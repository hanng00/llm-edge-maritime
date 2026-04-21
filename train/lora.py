"""LoRA fine-tuning for Gemma 3 4B.

Uses QLoRA (4-bit quantization) to fit on consumer GPUs.
"""

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


def format_training_example(scenario: dict, decision: dict) -> str:
    """Format a scenario + decision as a training example."""
    
    # Format scenario as concise text
    vessels_text = []
    for i, v in enumerate(scenario["vessels"], 1):
        ais = "AIS" if v["ais_active"] else "NO_AIS"
        vessels_text.append(
            f"C{i}: {v['vessel_type']} {v['bearing']:.0f}°/{v['distance']:.1f}nm "
            f"{v['speed']:.0f}kn {ais}"
        )
    
    prompt = f"""Mission: {scenario['mission_type']}
Own: {scenario['own_heading']:.0f}°/{scenario['own_speed']:.0f}kn
Conditions: {scenario['weather']}, vis:{scenario['visibility']}, {scenario['time_of_day']}, comms:{scenario['comms_status']}
Contacts:
{chr(10).join(vessels_text)}

Assess threat and recommend action."""

    response = json.dumps({
        "threat_level": decision["threat_level"],
        "action": decision["action"],
        "reasoning": decision["reasoning"],
        "confidence": decision["confidence"]
    })
    
    return prompt, response


def prepare_dataset(data_path: str, output_dir: str):
    """Prepare dataset for fine-tuning."""
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    
    # 80/20 split
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    console.print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Format for training
    for split_name, split_data in [("train", train_data), ("eval", eval_data)]:
        examples = []
        for item in split_data:
            prompt, response = format_training_example(item["scenario"], item["decision"])
            examples.append({
                "prompt": prompt,
                "response": response,
                "scenario_id": item["scenario"]["id"]
            })
        
        out_path = output / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        console.print(f"Wrote {len(examples)} examples to {out_path}")


@click.group()
def cli():
    """LoRA fine-tuning tools."""
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="data/prepared", help="Output directory")
def prepare(data_path: str, output: str):
    """Prepare dataset for fine-tuning."""
    prepare_dataset(data_path, output)


@cli.command()
@click.option("--base", default="google/gemma-3-4b-it", help="Base model")
@click.option("--data", default="data/prepared", help="Prepared data directory")
@click.option("--output", "-o", default="outputs/gemma-lora", help="Output directory")
@click.option("--epochs", default=3, help="Training epochs")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--lr", default=2e-4, help="Learning rate")
@click.option("--lora-r", default=16, help="LoRA rank")
@click.option("--lora-alpha", default=32, help="LoRA alpha")
def train(base: str, data: str, output: str, epochs: int, batch_size: int, lr: float, lora_r: int, lora_alpha: int):
    """Fine-tune model with LoRA.
    
    Requires: pip install torch transformers peft bitsandbytes datasets
    """
    console.print(f"[bold]LoRA Fine-tuning[/bold]")
    console.print(f"Base model: {base}")
    console.print(f"Data: {data}")
    console.print(f"Output: {output}")
    console.print(f"Config: epochs={epochs}, batch={batch_size}, lr={lr}, r={lora_r}, alpha={lora_alpha}")
    
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: uv pip install torch transformers peft bitsandbytes datasets")
        return
    
    # QLoRA config (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    console.print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": f"{data}/train.jsonl",
        "eval": f"{data}/eval.jsonl",
    })
    
    def tokenize(example):
        text = f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"
        return tokenizer(text, truncation=True, max_length=512, padding="max_length")
    
    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    
    # Training
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
    )
    
    console.print("Starting training...")
    trainer.train()
    
    console.print(f"Saving to {output}...")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    
    console.print("[green]Training complete![/green]")


if __name__ == "__main__":
    cli()
