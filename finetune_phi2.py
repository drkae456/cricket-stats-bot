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
        """Create training examples from the loaded data"""
        print("Preparing training data...")
        training_examples = []
        
        # Create examples for each data type
        for data_type, mapping in self.data.items():
            # Create general information about the data
            overview = f"Here's information about cricket {data_type}:\n"
            for key, value in list(mapping.items())[:10]:  # Sample first 10 items
                overview += f"- {key}: ID {value}\n"
            overview += f"There are {len(mapping)} {data_type} entries in total."
            
            training_examples.append({
                "instruction": f"Provide an overview of cricket {data_type}.",
                "input": "",
                "output": overview
            })
            
            # Create examples for specific lookups
            for key, value in list(mapping.items())[:50]:  # Sample items for training
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
        
        # Add cricket statistics examples if cricket_df is loaded
        if self.cricket_df is not None:
            training_examples.extend(self._generate_cricket_data_examples())
            training_examples.extend(self._generate_analytical_examples())
        
        # Add examples for cricket rules and terminology
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
            },
            {
                "instruction": "What is the difference between Test cricket and ODI?",
                "input": "",
                "output": "Test cricket and One Day Internationals (ODIs) differ in several ways. Test matches last up to 5 days with unlimited overs and players wear white clothing. ODIs are limited to 50 overs per team and are completed in a single day, with players wearing colored uniforms. Test cricket is considered the traditional format that tests players' endurance and technique, while ODIs are more fast-paced with an emphasis on scoring quickly."
            },
            {
                "instruction": "What is a hat-trick in cricket?",
                "input": "",
                "output": "A hat-trick in cricket occurs when a bowler dismisses three batters with consecutive deliveries. This is a rare and celebrated achievement. The term originated in cricket but has since been adopted by other sports. In international cricket, hat-tricks are relatively uncommon and are considered a significant milestone in a bowler's career."
            }
        ]
        training_examples.extend(cricket_rules)
        
        # Add examples for cricket formats
        if 'format' in self.data:
            format_examples = [
                {
                    "instruction": "What are the different formats of cricket?",
                    "input": "",
                    "output": f"The main formats of cricket are Test cricket (played over 5 days), One Day Internationals (50 overs per side), and Twenty20 (20 overs per side). According to our data, we have information on {', '.join(list(self.data['format'].keys()))}."
                },
                {
                    "instruction": "How long does a Test match last?",
                    "input": "",
                    "output": "A Test match in cricket is scheduled to last up to 5 days, with each day typically consisting of 90 overs (6 hours of play). It's the longest format of cricket and is considered the ultimate test of a player's skill and endurance. Test matches can end earlier if a result is achieved before the 5 days are complete."
                }
            ]
            training_examples.extend(format_examples)
        
        # Add examples for dismissal types
        if 'dismissal' in self.data:
            dismissal_examples = [
                {
                    "instruction": "What are the different ways a batter can get out in cricket?",
                    "input": "",
                    "output": f"In cricket, a batter can be dismissed in several ways including: {', '.join(list(self.data['dismissal'].keys()))}. The most common dismissals are caught, bowled, and LBW (Leg Before Wicket)."
                }
            ]
            training_examples.extend(dismissal_examples)
        
        print(f"Created {len(training_examples)} training examples")
        return training_examples
    
    def _generate_cricket_data_examples(self):
        """Generate examples from the cricket data CSV"""
        examples = []
        
        # Create reverse mappings for easier lookup
        player_id_to_name = {v: k for k, v in self.data.get('player', {}).items()}
        team_id_to_name = {v: k for k, v in self.data.get('team', {}).items()}
        stadium_id_to_name = {v: k for k, v in self.data.get('stadium', {}).items()}
        
        # Add examples for player batting statistics
        if 'batter' in self.cricket_df.columns and 'runs_batter' in self.cricket_df.columns:
            sample_batters = self.cricket_df['batter'].unique()[:20]  # Sample 20 batters
            for batter_id in sample_batters:
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
        
        # Add examples for team performance
        if 'batting_team' in self.cricket_df.columns and 'bowling_team' in self.cricket_df.columns:
            sample_teams = self.cricket_df['batting_team'].unique()[:10]  # Sample 10 teams
            for team_id in sample_teams:
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
        
        return examples
    
    def _generate_analytical_examples(self):
        """Generate analytical and predictive examples from the cricket data"""
        examples = []
        
        # Create reverse mappings for easier lookup
        player_id_to_name = {v: k for k, v in self.data.get('player', {}).items()}
        team_id_to_name = {v: k for k, v in self.data.get('team', {}).items()}
        stadium_id_to_name = {v: k for k, v in self.data.get('stadium', {}).items()}
        
        # Check if we have the necessary columns for analysis
        required_columns = ['match_id', 'batting_team', 'bowling_team', 'stadium', 'runs_batter']
        if not all(col in self.cricket_df.columns for col in required_columns):
            print("Warning: Missing required columns for analytical examples")
            return examples
        
        # 1. Team vs Team at specific venue analysis
        # Find teams that have played against each other at the same venue multiple times
        team_venue_matches = {}
        
        for _, row in self.cricket_df.drop_duplicates(['match_id', 'batting_team', 'bowling_team', 'stadium']).iterrows():
            match_id = row['match_id']
            batting_team = row['batting_team']
            bowling_team = row['bowling_team']
            stadium = row['stadium']
            
            # Create a unique key for this team-vs-team at venue combination
            key = (min(batting_team, bowling_team), max(batting_team, bowling_team), stadium)
            
            if key not in team_venue_matches:
                team_venue_matches[key] = []
            
            team_venue_matches[key].append(match_id)
        
        # Filter to combinations with at least 2 matches
        team_venue_analysis = {k: v for k, v in team_venue_matches.items() if len(v) >= 2}
        
        # Generate analytical examples for these combinations
        for (team1_id, team2_id, stadium_id), match_ids in list(team_venue_analysis.items())[:5]:  # Take first 5 examples
            if team1_id in team_id_to_name and team2_id in team_id_to_name and stadium_id in stadium_id_to_name:
                team1_name = team_id_to_name[team1_id]
                team2_name = team_id_to_name[team2_id]
                stadium_name = stadium_id_to_name[stadium_id]
                
                # Get match data
                match_data = self.cricket_df[self.cricket_df['match_id'].isin(match_ids)]
                
                # Calculate team1 batting stats
                team1_batting = match_data[match_data['batting_team'] == team1_id]
                team1_runs = team1_batting['runs_batter'].sum()
                
                # Calculate team2 batting stats
                team2_batting = match_data[match_data['batting_team'] == team2_id]
                team2_runs = team2_batting['runs_batter'].sum()
                
                # Determine which team performed better
                if team1_runs > team2_runs:
                    better_team = team1_name
                    better_runs = team1_runs
                    worse_team = team2_name
                    worse_runs = team2_runs
                else:
                    better_team = team2_name
                    better_runs = team2_runs
                    worse_team = team1_name
                    worse_runs = team1_runs
                
                # Create analytical example
                analysis = f"Based on historical data of {len(match_ids)} matches between {team1_name} and {team2_name} at {stadium_name}:\n\n"
                analysis += f"1. {team1_name} has scored a total of {team1_runs} runs\n"
                analysis += f"2. {team2_name} has scored a total of {team2_runs} runs\n\n"
                analysis += f"Overall, {better_team} has performed better at this venue with {better_runs} runs compared to {worse_team}'s {worse_runs} runs.\n\n"
                
                if 'date' in self.cricket_df.columns:
                    # Sort matches by date to analyze trends
                    match_dates = match_data.drop_duplicates('match_id')[['match_id', 'date']].sort_values('date')
                    analysis += f"Recent trend: In the most recent match on {match_dates.iloc[-1]['date']}, "
                    
                    recent_match = match_data[match_data['match_id'] == match_dates.iloc[-1]['match_id']]
                    recent_team1_runs = recent_match[recent_match['batting_team'] == team1_id]['runs_batter'].sum()
                    recent_team2_runs = recent_match[recent_match['batting_team'] == team2_id]['runs_batter'].sum()
                    
                    if recent_team1_runs > recent_team2_runs:
                        analysis += f"{team1_name} outperformed {team2_name}.\n\n"
                    else:
                        analysis += f"{team2_name} outperformed {team1_name}.\n\n"
                
                analysis += f"Prediction: Based on historical performance, {better_team} has an advantage at {stadium_name} against {worse_team}."
                
                examples.append({
                    "instruction": f"Analyze the historical performance of {team1_name} vs {team2_name} at {stadium_name} and predict the outcome of their next match.",
                    "input": "",
                    "output": analysis
                })
        
        # 2. Player performance at specific venues
        if 'batter' in self.cricket_df.columns:
            player_venue_data = {}
            
            for batter_id in self.cricket_df['batter'].unique():
                if batter_id in player_id_to_name:
                    batter_name = player_id_to_name[batter_id]
                    batter_data = self.cricket_df[self.cricket_df['batter'] == batter_id]
                    
                    # Group by stadium
                    for stadium_id, stadium_data in batter_data.groupby('stadium'):
                        if stadium_id in stadium_id_to_name:
                            stadium_name = stadium_id_to_name[stadium_id]
                            
                            # Only consider venues where player has played multiple innings
                            if len(stadium_data) >= 3:
                                key = (batter_id, stadium_id)
                                player_venue_data[key] = {
                                    'name': batter_name,
                                    'venue': stadium_name,
                                    'innings': len(stadium_data),
                                    'runs': stadium_data['runs_batter'].sum(),
                                    'avg_runs': stadium_data['runs_batter'].mean()
                                }
            
            # Generate examples for player venue analysis
            for (player_id, stadium_id), stats in list(player_venue_data.items())[:5]:  # Take first 5 examples
                player_name = stats['name']
                stadium_name = stats['venue']
                
                analysis = f"Analysis of {player_name}'s performance at {stadium_name}:\n\n"
                analysis += f"1. {player_name} has played {stats['innings']} innings at this venue\n"
                analysis += f"2. Total runs scored: {stats['runs']}\n"
                analysis += f"3. Average runs per innings: {stats['avg_runs']:.2f}\n\n"
                
                # Compare with overall average
                all_innings = self.cricket_df[self.cricket_df['batter'] == player_id]
                overall_avg = all_innings['runs_batter'].mean()
                
                if stats['avg_runs'] > overall_avg:
                    analysis += f"{player_name} performs better at {stadium_name} compared to their overall average of {overall_avg:.2f} runs per innings.\n\n"
                    analysis += f"Prediction: {player_name} is likely to perform well in their next match at {stadium_name}."
                else:
                    analysis += f"{player_name} performs worse at {stadium_name} compared to their overall average of {overall_avg:.2f} runs per innings.\n\n"
                    analysis += f"Prediction: {player_name} might struggle in their next match at {stadium_name}."
                
                examples.append({
                    "instruction": f"Analyze {player_name}'s performance at {stadium_name} and predict how they might perform in their next match there.",
                    "input": "",
                    "output": analysis
                })
        
        # 3. Team form analysis
        if 'date' in self.cricket_df.columns and 'match_id' in self.cricket_df.columns:
            # Get recent matches for analysis
            recent_matches = self.cricket_df.sort_values('date', ascending=False).drop_duplicates('match_id')
            
            # Analyze recent form of teams
            for team_id in self.cricket_df['batting_team'].unique()[:5]:  # Take first 5 teams
                if team_id in team_id_to_name:
                    team_name = team_id_to_name[team_id]
                    
                    # Get matches where this team batted
                    team_matches = recent_matches[
                        (recent_matches['batting_team'] == team_id) | 
                        (recent_matches['bowling_team'] == team_id)
                    ]
                    
                    if len(team_matches) >= 3:  # Need at least 3 matches for form analysis
                        recent_3_matches = team_matches.head(3)
                        
                        analysis = f"Recent form analysis for {team_name}:\n\n"
                        
                        # Analyze each match
                        for i, (_, match) in enumerate(recent_3_matches.iterrows(), 1):
                            match_id = match['match_id']
                            match_date = match['date']
                            
                            # Get full match data
                            full_match = self.cricket_df[self.cricket_df['match_id'] == match_id]
                            
                            # Get opponent
                            if match['batting_team'] == team_id:
                                opponent_id = match['bowling_team']
                            else:
                                opponent_id = match['batting_team']
                            
                            if opponent_id in team_id_to_name:
                                opponent_name = team_id_to_name[opponent_id]
                                
                                # Calculate runs
                                team_runs = full_match[full_match['batting_team'] == team_id]['runs_batter'].sum()
                                opponent_runs = full_match[full_match['batting_team'] == opponent_id]['runs_batter'].sum()
                                
                                result = "won" if team_runs > opponent_runs else "lost"
                                
                                analysis += f"Match {i} ({match_date}): {team_name} {result} against {opponent_name} "
                                analysis += f"({team_runs} runs vs {opponent_runs} runs)\n"
                        
                        # Overall form assessment
                        wins = sum(1 for _, match in recent_3_matches.iterrows() 
                                  if self.cricket_df[(self.cricket_df['match_id'] == match['match_id']) & 
                                                   (self.cricket_df['batting_team'] == team_id)]['runs_batter'].sum() > 
                                     self.cricket_df[(self.cricket_df['match_id'] == match['match_id']) & 
                                                   (self.cricket_df['batting_team'] != team_id)]['runs_batter'].sum())
                        
                        if wins >= 2:
                            analysis += f"\nOverall: {team_name} is in good form, winning {wins} of their last 3 matches."
                        else:
                            analysis += f"\nOverall: {team_name} is struggling in recent matches, winning only {wins} of their last 3 matches."
                        
                        examples.append({
                            "instruction": f"Analyze the recent form of {team_name} based on their last few matches.",
                            "input": "",
                            "output": analysis
                        })
        
        # 4. Specific match prediction examples
        # These are template examples to teach the model how to make predictions
        prediction_templates = [
            {
                "instruction": "New Zealand are playing against Australia at Melbourne Cricket Ground. What's the breakdown of the match based on past games, and predict the outcome based on recent form.",
                "input": "",
                "output": """Based on historical data between New Zealand and Australia at Melbourne Cricket Ground:

1. Head-to-head at this venue: Australia has won 7 out of 10 matches against New Zealand at MCG.
2. Average first innings score at MCG: 285 runs
3. Average second innings score at MCG: 245 runs
4. Toss impact: Teams winning the toss have won 60% of matches at this venue

Recent form:
- Australia: Won 4 of their last 5 matches, showing strong batting performances
- New Zealand: Won 3 of their last 5 matches, with their bowling attack performing exceptionally well

Key player analysis:
- Kane Williamson (NZ) averages 48.5 runs at MCG
- Mitchell Starc (AUS) has taken 15 wickets in his last 5 matches at this venue

Prediction:
Australia has a historical advantage at MCG and their recent form suggests they are slight favorites. However, New Zealand's bowling attack could pose challenges. If New Zealand bats first and scores over 300, they have a good chance of winning. Otherwise, Australia is likely to win by a margin of 30-40 runs or 5-6 wickets."""
            },
            {
                "instruction": "India vs England at Lord's Cricket Ground. Analyze past performances and predict the outcome.",
                "input": "",
                "output": """Analysis of India vs England at Lord's Cricket Ground:

Historical performance:
1. England has won 12 matches against India at Lord's, while India has won 3
2. India's last win at Lord's came in 2021, breaking a 7-match losing streak at this venue
3. Average first innings score: 320 runs
4. Average fourth innings chase: 180 runs (highest successful chase: 282)

Recent form:
- India: Won 7 of their last 10 Test matches, showing excellent batting depth
- England: Won 5 of their last 10 Test matches, with their "Bazball" approach yielding mixed results

Key player analysis:
- Virat Kohli averages 36.4 at Lord's with one century
- James Anderson has taken 115 wickets at Lord's, his most successful venue

Weather and pitch conditions:
- Forecast suggests overcast conditions for the first two days, favoring swing bowling
- The pitch traditionally offers movement on days 1-2, then flattens out, before deteriorating on day 5

Prediction:
Given the conditions and recent form, this match will likely be closely contested. England has the historical advantage, but India's recent improvement in overseas conditions makes them competitive. If conditions remain overcast, England's pace attack gives them a 60% chance of winning. In drier conditions, India's balanced attack and strong batting gives them a slight edge."""
            }
        ]
        examples.extend(prediction_templates)
        
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
    
    def train(self, training_examples, output_dir="./cricket_model", epochs=3, batch_size=8):
        """Train the model on the provided examples"""
        print(f"Preparing to train with {len(training_examples)} examples...")
        
        # Create dataset
        train_dataset = Dataset.from_list(training_examples)
        train_dataset = train_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,  # Use BF16 for H100
            tf32=True,  # Enable TF32 for faster matrix multiplications
            remove_unused_columns=False,
            optim="adamw_torch_fused"
        )
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model
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
    cricket_model.train(training_examples)
    
    # Start interactive mode
    cricket_model.interactive_mode()

if __name__ == "__main__":
    main()
