import json
import os
import pandas as pd
import numpy as np
import torch
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

class CricketAnalysisModel:
    def __init__(self, model_name="microsoft/phi-2", lora_r=16, lora_alpha=32, lora_dropout=0.05):
        self.model_name = model_name
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.tokenizer = None
        self.model = None
        self.data = {}
        self.cricket_df = None
        
    def extract_zip_data(self, zip_path="cleaned_data.zip", extract_to="./"):
        """Extract the zip file containing the data"""
        print(f"Extracting data from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Data extraction complete")
        
    def load_data(self, data_dir="cleaned_data"):
        """Load all JSON mapping files and CSV data from the data directory"""
        print("Loading data...")
        
        # Only load the mapping files that exist in the data directory
        valid_mappings = [
            "city", "dismissal", "extras_type", "format", 
            "player", "stadium", "team", "tournament"
        ]
        
        # Load JSON mapping files
        for mapping_type in valid_mappings:
            filename = f"{mapping_type}_mapping.json"
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data[mapping_type] = json.load(f)
                    print(f"Loaded {mapping_type} mapping with {len(self.data[mapping_type])} entries")
            else:
                print(f"Warning: Could not find {filename}")
        
        # Load cricket CSV data
        csv_path = os.path.join(data_dir, "cleaned_cricket_data.csv")
        if os.path.exists(csv_path):
            self.cricket_df = pd.read_csv(csv_path)
            print(f"Loaded cricket data with {len(self.cricket_df)} rows")
        else:
            # Check if it's in a nested directory
            nested_csv_path = os.path.join(data_dir, "cleaned_data", "cleaned_cricket_data.csv")
            if os.path.exists(nested_csv_path):
                self.cricket_df = pd.read_csv(nested_csv_path)
                print(f"Loaded cricket data with {len(self.cricket_df)} rows")
            else:
                print(f"Warning: Could not find cricket data CSV at {csv_path} or {nested_csv_path}")
        
        print(f"Loaded {len(self.data)} mapping files")
        
    def prepare_training_data(self):
        """Create training examples from the loaded data using a tiny subset for testing"""
        print("Preparing training data (tiny subset for testing)...")
        training_examples = []
        
        # Create examples for each data type - limit to just 2 mappings
        selected_mappings = list(self.data.keys())[:2]  # Only use first 2 mapping types
        for data_type in selected_mappings:
            mapping = self.data[data_type]
            # Take only 5 items from each mapping
            sample_mapping = dict(list(mapping.items())[:5])
            
            # Create general information about the data
            overview = f"Here's information about cricket {data_type}:\n"
            for key, value in sample_mapping.items():
                overview += f"- {key}: ID {value}\n"
            overview += f"There are {len(mapping)} {data_type} entries in total."
            
            training_examples.append({
                "instruction": f"Provide an overview of cricket {data_type}.",
                "input": "",
                "output": overview
            })
            
            # Create examples for specific lookups - use only the sample items
            for key, value in sample_mapping.items():
                training_examples.append({
                    "instruction": f"What is the ID for {key} in {data_type}?",
                    "input": "",
                    "output": f"The ID for {key} in {data_type} is {value}."
                })
                
                training_examples.append({
                    "instruction": f"What {data_type} corresponds to ID {value}?",
                    "input": "",
                    "output": f"ID {value} in {data_type} corresponds to {key}."
                })
        
        # Add cricket statistics examples if cricket_df is loaded - limit to 10 examples
        if self.cricket_df is not None:
            cricket_examples = self._generate_cricket_data_examples()
            training_examples.extend(cricket_examples[:10])  # Only use first 10 examples
        
        # Add just 3 examples for cricket rules and terminology
        cricket_rules = [
            {
                "instruction": "Explain the LBW rule in cricket.",
                "input": "",
                "output": "LBW (Leg Before Wicket) is a method of dismissal in cricket. A batter is out LBW if the ball would have hit the stumps but was intercepted by any part of the batter's body (except the hand holding the bat). The ball must not pitch outside the leg stump, and the impact must either be in line with the stumps or outside the off stump if the batter is not playing a shot."
            },
            {
                "instruction": "What is a yorker in cricket?",
                "input": "",
                "output": "A yorker is a type of delivery in cricket where the ball lands directly at the batter's feet, making it difficult to hit. It's considered one of the most effective deliveries, especially in limited-overs cricket, as it can either result in the batter being bowled or trapped LBW."
            },
            {
                "instruction": "Explain what a duck means in cricket.",
                "input": "",
                "output": "In cricket, a 'duck' refers to when a batter is dismissed without scoring any runs (zero). The term comes from the shape of the number '0' resembling a duck's egg. A 'golden duck' specifically means the batter was dismissed on the first ball they faced."
            }
        ]
        training_examples.extend(cricket_rules)
        
        print(f"Created {len(training_examples)} training examples (tiny subset)")
        return training_examples
    
    def _generate_cricket_data_examples(self):
        """Generate examples from the cricket data CSV"""
        examples = []
        
        # Convert player IDs to names
        player_mapping = json.load(open('player_mapping.json'))
        player_id_to_name = {v: k for k, v in player_mapping.items()}
        self.cricket_df['batter_name'] = self.cricket_df['batter'].map(player_id_to_name)
        
        # Create reverse mappings for easier lookup
        team_id_to_name = {v: k for k, v in self.data.get('team', {}).items()}
        stadium_id_to_name = {v: k for k, v in self.data.get('stadium', {}).items()}
        
        # Add examples for player batting statistics - use all batters instead of just 20
        if 'batter' in self.cricket_df.columns and 'runs_batter' in self.cricket_df.columns:
            all_batters = self.cricket_df['batter'].unique()
            for batter_id in all_batters:
                if batter_id in player_id_to_name:
                    batter_name = player_id_to_name[batter_id]
                    batter_data = self.cricket_df[self.cricket_df['batter'] == batter_id]
                    
                    # Calculate batting stats
                    total_runs = batter_data['runs_batter'].sum()
                    innings = len(batter_data['innings'].unique()) if 'innings' in self.cricket_df.columns else len(batter_data)
                    
                    examples.append({
                        "instruction": f"How many runs has {batter_name} scored in total?",
                        "input": "",
                        "output": f"{batter_name} has scored a total of {total_runs} runs in {innings} innings based on the available data."
                    })
                    
                    # Add more detailed batting analysis
                    if 'balls_faced' in self.cricket_df.columns:
                        total_balls = batter_data['balls_faced'].sum()
                        strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
                        
                        examples.append({
                            "instruction": f"What is {batter_name}'s strike rate?",
                            "input": "",
                            "output": f"{batter_name}'s strike rate is {strike_rate:.2f} (scored {total_runs} runs from {total_balls} balls)."
                        })
        
        # Add examples for team performance - use all teams instead of just 10
        if 'batting_team' in self.cricket_df.columns and 'bowling_team' in self.cricket_df.columns:
            all_teams = self.cricket_df['batting_team'].unique()
            for team_id in all_teams:
                if team_id in team_id_to_name:
                    team_name = team_id_to_name[team_id]
                    team_batting = self.cricket_df[self.cricket_df['batting_team'] == team_id]
                    team_bowling = self.cricket_df[self.cricket_df['bowling_team'] == team_id]
                    
                    # Calculate team stats
                    matches_batted = len(team_batting['match_id'].unique()) if 'match_id' in self.cricket_df.columns else 'multiple'
                    matches_bowled = len(team_bowling['match_id'].unique()) if 'match_id' in self.cricket_df.columns else 'multiple'
                    
                    examples.append({
                        "instruction": f"How many matches has {team_name} played?",
                        "input": "",
                        "output": f"Based on the available data, {team_name} has batted in {matches_batted} matches and bowled in {matches_bowled} matches."
                    })
                    
                    # Add team runs information
                    total_runs = team_batting['runs_batter'].sum() if 'runs_batter' in self.cricket_df.columns else 'unknown'
                    examples.append({
                        "instruction": f"How many total runs has {team_name} scored?",
                        "input": "",
                        "output": f"{team_name} has scored a total of {total_runs} runs in the available data."
                    })
                    
                    # Add team wickets information if available
                    if 'wicket' in self.cricket_df.columns:
                        wickets_taken = team_bowling[team_bowling['wicket'] == 1].shape[0]
                        examples.append({
                            "instruction": f"How many wickets has {team_name} taken?",
                            "input": "",
                            "output": f"{team_name} has taken {wickets_taken} wickets according to the available data."
                        })
        
        # Add examples for stadium statistics - use all stadiums instead of just 10
        if 'stadium' in self.cricket_df.columns:
            all_stadiums = self.cricket_df['stadium'].unique()
            for stadium_id in all_stadiums:
                if stadium_id in stadium_id_to_name:
                    stadium_name = stadium_id_to_name[stadium_id]
                    stadium_data = self.cricket_df[self.cricket_df['stadium'] == stadium_id]
                    
                    # Calculate stadium stats
                    matches = len(stadium_data['match_id'].unique()) if 'match_id' in self.cricket_df.columns else 'multiple'
                    
                    examples.append({
                        "instruction": f"How many matches have been played at {stadium_name}?",
                        "input": "",
                        "output": f"According to the available data, {matches} matches have been played at {stadium_name}."
                    })
                    
                    # Add average score information if available
                    if 'runs_batter' in self.cricket_df.columns and 'innings' in self.cricket_df.columns:
                        innings_data = stadium_data.groupby('innings')['runs_batter'].sum()
                        if not innings_data.empty:
                            avg_score = innings_data.mean()
                            examples.append({
                                "instruction": f"What is the average score at {stadium_name}?",
                                "input": "",
                                "output": f"The average score per innings at {stadium_name} is {avg_score:.2f} runs based on the available data."
                            })
        
        # Generate examples for all available columns in the dataset
        # First, identify numerical columns that might contain statistics
        numerical_cols = self.cricket_df.select_dtypes(include=['number']).columns
        
        # For each player, generate examples about their statistics in each numerical column
        for batter_id in self.cricket_df['batter'].unique():
            if batter_id in player_id_to_name:
                batter_name = player_id_to_name[batter_id]
                batter_data = self.cricket_df[self.cricket_df['batter'] == batter_id]
                
                for col in numerical_cols:
                    if col in ['batter', 'bowler', 'match_id', 'innings', 'batting_team', 'bowling_team', 'stadium']:
                        continue  # Skip ID columns
                    
                    # Only create examples for columns that have meaningful data for this player
                    if batter_data[col].sum() > 0:
                        col_total = batter_data[col].sum()
                        col_avg = batter_data[col].mean()
                        
                        # Format column name for readability
                        readable_col = col.replace('_', ' ')
                        
                        examples.append({
                            "instruction": f"What is {batter_name}'s total {readable_col}?",
                            "input": "",
                            "output": f"{batter_name}'s total {readable_col} is {col_total}."
                        })
                        
                        examples.append({
                            "instruction": f"What is {batter_name}'s average {readable_col}?",
                            "input": "",
                            "output": f"{batter_name}'s average {readable_col} is {col_avg:.2f}."
                        })
        
        # Create player summary statistics
        player_stats = {}
        for player_id, player_name in player_id_to_name.items():
            player_data = self.cricket_df[self.cricket_df['batter'] == player_id]
            player_stats[player_name] = {
                'total_runs': player_data['runs_batter'].sum(),
                'matches': player_data['date'].nunique(),
                'venues': player_data['venue'].nunique(),
                # Add more statistics as needed
            }
        
        return examples
    
    def initialize_model(self):
        """Initialize the model and tokenizer"""
        print(f"Initializing model {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token to be the same as EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        print(f"Model initialized with {self.lora_config.r} LoRA rank")
    
    def format_prompt(self, instruction, input_text="", output=None):
        """Format the prompt for the model using phi-2 chat format"""
        if input_text:
            prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endoftext|>\n<|assistant|>"
        else:
            prompt = f"<|user|>\n{instruction}<|endoftext|>\n<|assistant|>"
        
        # If output is provided, add it (for training)
        if output is not None:
            prompt += f"{output}<|endoftext|>"
        
        return prompt
    
    def preprocess_data(self, examples):
        """Process the data into model inputs with proper labels"""
        prompts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = self.format_prompt(instruction, input_text, output)
            prompts.append(prompt)
        
        # Tokenize the prompts
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self, output_dir="cricket_model", num_train_epochs=1, per_device_train_batch_size=4):
        """Train the model with a tiny subset of data and minimal training"""
        print("Starting training with minimal parameters for testing...")
        
        # Prepare training data
        training_data = self.prepare_training_data()
        train_dataset = Dataset.from_list(training_data)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            prompt = self.format_prompt(examples["instruction"], examples["input"], examples["output"])
            target = examples["output"]
            
            tokenized_prompt = self.tokenizer(prompt, truncation=True, max_length=512)
            tokenized_target = self.tokenizer(target, truncation=True, max_length=512)
            
            # Combine prompt and target for training
            input_ids = tokenized_prompt["input_ids"] + tokenized_target["input_ids"][1:]  # Skip the BOS token
            attention_mask = tokenized_prompt["attention_mask"] + tokenized_target["attention_mask"][1:]
            
            # Create labels - set prompt tokens to -100 to ignore them in loss calculation
            labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_target["input_ids"][1:]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        tokenized_dataset = train_dataset.map(tokenize_function, remove_columns=["instruction", "input", "output"])
        
        # Set up training arguments - minimal settings for testing
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            max_steps=20,  # Limit to just 20 steps for quick testing
            learning_rate=1e-4,
            fp16=True,
            logging_steps=1,  # Log every step
            save_steps=10,  # Save only twice during training
            save_total_limit=2,  # Keep only 2 checkpoints
            report_to="none",  # Disable wandb or other reporting
        )
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def generate_response(self, instruction, input_text="", max_length=512):
        """Generate a response to a user query"""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("<|assistant|>")[-1].strip()
        return response
    
    def interactive_mode(self):
        """Run an interactive session with the model"""
        print("Starting interactive mode. Type 'exit' to quit.")
        while True:
            instruction = input("\nUser: ")
            if instruction.lower() == 'exit':
                break
            
            response = self.generate_response(instruction)
            print(f"\nAssistant: {response}")

def main():
    # Initialize the model
    cricket_model = CricketAnalysisModel()
    
    # Extract zip data if needed
    if os.path.exists("cleaned_data.zip"):
        cricket_model.extract_zip_data()
    
    # Load data
    cricket_model.load_data()
    
    # Prepare training data
    training_examples = cricket_model.prepare_training_data()
    
    # Initialize model
    cricket_model.initialize_model()
    
    # Train the model
    cricket_model.train()
    
    # Start interactive mode
    cricket_model.interactive_mode()

if __name__ == "__main__":
    main()
