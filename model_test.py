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
from mappings import cricket_mappings
from sentence_transformers import SentenceTransformer
import faiss

class CricketAnalysisModel:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.data = {}
        self.cricket_df = None
        
        # Fix the data directory path to point to the correct location
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, 'cleaned_data')
        
        # Define LoRA config
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Add vector database for document retrieval
        self.vector_db = None
        self.embeddings_model = None
        
    def extract_zip_data(self, zip_path="cleaned_data.zip", extract_to="./"):
        """Extract the zip file containing the data"""
        print(f"Extracting data from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Data extraction complete")
        
    def load_data(self):
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
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data[mapping_type] = json.load(f)
                    print(f"Loaded {mapping_type} mapping with {len(self.data[mapping_type])} entries")
            else:
                print(f"Warning: Could not find {filename}")
        
        # Load cricket CSV data
        csv_path = os.path.join(self.data_dir, "cleaned_cricket_data.csv")
        if os.path.exists(csv_path):
            self.cricket_df = pd.read_csv(csv_path)
            print(f"Loaded cricket data with {len(self.cricket_df)} rows")
        else:
            # Check if it's in a nested directory
            nested_csv_path = os.path.join(self.data_dir, "cleaned_data", "cleaned_cricket_data.csv")
            if os.path.exists(nested_csv_path):
                self.cricket_df = pd.read_csv(nested_csv_path)
                print(f"Loaded cricket data with {len(self.cricket_df)} rows")
            else:
                print(f"Warning: Could not find cricket data CSV at {csv_path} or {nested_csv_path}")
        
        print(f"Loaded {len(self.data)} mapping files")
        
    def prepare_training_data(self):
        """Prepare training data from various sources"""
        print("Preparing training data...")
        
        training_data = []
        
        # Load any additional data files needed for training
        if os.path.exists(os.path.join(self.data_dir, 'additional_training_data.json')):
            with open(os.path.join(self.data_dir, 'additional_training_data.json'), 'r') as f:
                additional_data = json.load(f)
                training_data.extend(additional_data)
        
        # Generate examples from cricket data
        cricket_examples = self._generate_cricket_data_examples()
        training_data.extend(cricket_examples)
        
        # Remove the call to _generate_synthetic_examples since it doesn't exist
        # Or implement a basic version:
        # synthetic_examples = self._generate_synthetic_examples()
        # training_data.extend(synthetic_examples)
        
        print(f"Total training examples: {len(training_data)}")
        return training_data
    
    def _generate_cricket_data_examples(self):
        """Generate examples from cricket data"""
        print("Generating cricket data examples...")
        
        # Load mapping files
        player_mapping = json.load(open(os.path.join(self.data_dir, 'player_mapping.json')))
        team_mapping = json.load(open(os.path.join(self.data_dir, 'team_mapping.json')))
        stadium_mapping = json.load(open(os.path.join(self.data_dir, 'stadium_mapping.json')))
        
        # Load the cleaned cricket data
        cricket_data = pd.read_csv(os.path.join(self.data_dir, 'cleaned_cricket_data.csv'))
        print(f"Loaded cricket data with {len(cricket_data)} rows")
        
        examples = []
        
        # Group data by match (date, venue, teams)
        match_groups = cricket_data.groupby(['date', 'venue', 'batting_team', 'bowling_team'])
        
        for (date, venue, batting_team_id, bowling_team_id), match_data in match_groups:
            try:
                # Get team names from IDs
                batting_team = next((k for k, v in team_mapping.items() if v == batting_team_id), f"Team {batting_team_id}")
                bowling_team = next((k for k, v in team_mapping.items() if v == bowling_team_id), f"Team {bowling_team_id}")
                
                # Get venue name from ID
                venue_name = next((k for k, v in stadium_mapping.items() if v == venue), f"Venue {venue}")
                
                # Create match summary
                match_summary = f"{batting_team} vs {bowling_team} at {venue_name} on {date}"
                
                # Generate examples for this match
                batters_runs = match_data.groupby('batter')['runs_batter'].sum().reset_index()
                if not batters_runs.empty:
                    top_batter_id = batters_runs.loc[batters_runs['runs_batter'].idxmax(), 'batter']
                    top_batter = next((k for k, v in player_mapping.items() if v == top_batter_id), f"Player {top_batter_id}")
                    top_runs = batters_runs.loc[batters_runs['runs_batter'].idxmax(), 'runs_batter']
                    
                    question = f"Who was the top scorer in the match between {batting_team} and {bowling_team} on {date}?"
                    answer = f"The top scorer was {top_batter} with {top_runs} runs."
                    
                    examples.append({
                        "instruction": question,
                        "input": match_summary,
                        "output": answer
                    })
            except Exception as e:
                # Skip this match if there's an error
                print(f"Error processing match on {date}: {e}")
                continue
        
        print(f"Generated {len(examples)} examples from cricket data")
        return examples
    
    def initialize_model(self):
        """Initialize the model and tokenizer"""
        print(f"Initializing model {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token to be the same as EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # RTX 4090 has 24GB VRAM - we can use 8-bit quantization for better quality
        # while still having enough memory for decent batch sizes
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better RTX compatibility
            device_map="auto",
            load_in_8bit=True,  # 8-bit quantization offers better quality than 4-bit
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA with slightly higher rank for RTX 4090
        self.lora_config.r = 32  # Increase LoRA rank from 16 to 32 for better model quality
        self.model = get_peft_model(self.model, self.lora_config)
        
        print(f"Model initialized with {self.lora_config.r} LoRA rank using 8-bit quantization")
    
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
    
    def train(self):
        """Train the model on the prepared data"""
        print("Starting training with RTX 4090 optimized parameters...")
        
        # Prepare training data
        training_data = self.prepare_training_data()
        
        # Convert to dataset format
        dataset_dict = {
            "instruction": [item["instruction"] for item in training_data],
            "input": [item["input"] for item in training_data],
            "output": [item["output"] for item in training_data]
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # Process the dataset
        def preprocess_function(examples):
            inputs = []
            targets = []
            
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                # Format the prompt
                prompt = self.format_prompt(instruction, input_text)
                inputs.append(prompt)
                targets.append(output)
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None  # Return Python lists
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None  # Return Python lists
            )
            
            # Create the final labels
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Adjusted for RTX 4090
            gradient_accumulation_steps=4,  # Accumulate gradients for effective batch size of 16
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=200,
            fp16=True,  # Use mixed precision for faster training
            learning_rate=2e-5,
            remove_unused_columns=False,
        )
        
        # Create data collator that handles padding
        data_collator = lambda data: {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in data]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in data]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in data]),
        }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        self.model.save_pretrained("./cricket_model")
        self.tokenizer.save_pretrained("./cricket_model")
        print("Model training complete and saved to ./cricket_model")
    
    def generate_response(self, instruction, input_text="", max_length=512):
        """Generate a response to a user query with RTX 4090 optimized settings"""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,  # Add repetition penalty for better outputs
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("<|assistant|>")[-1].strip()
        return response
    
    def setup_retrieval_system(self, embeddings_model="sentence-transformers/all-mpnet-base-v2"):
        """Set up the retrieval system with document embeddings"""
        print(f"Setting up retrieval system with {embeddings_model}...")
        
        # Initialize the embeddings model
        self.embeddings_model = SentenceTransformer(embeddings_model)
        
        # Create document store from cricket data
        self.index_documents()
        
        print("Retrieval system setup complete")
    
    def index_documents(self):
        """Index documents from cricket data for retrieval"""
        print("Indexing cricket documents...")
        
        documents = []
        document_embeddings = []
        
        # Create documents from cricket data
        if self.cricket_df is not None:
            # Create match summaries
            for _, match_group in self.cricket_df.groupby(['date', 'venue', 'batting_team', 'bowling_team']):
                # Create a document for this match
                match_doc = self._create_match_document(match_group)
                documents.append(match_doc)
        
        # Create documents from player profiles
        if 'player' in self.data:
            for player_name, player_id in self.data['player'].items():
                player_doc = self._create_player_document(player_name, player_id)
                documents.append(player_doc)
        
        # Create documents from team information
        if 'team' in self.data:
            for team_name, team_id in self.data['team'].items():
                team_doc = self._create_team_document(team_name, team_id)
                documents.append(team_doc)
        
        # Create embeddings for all documents
        print(f"Creating embeddings for {len(documents)} documents...")
        for doc in tqdm(documents):
            embedding = self.embeddings_model.encode(doc['content'])
            document_embeddings.append(embedding)
        
        # Create FAISS index
        dimension = len(document_embeddings[0])
        self.vector_db = faiss.IndexFlatL2(dimension)
        self.vector_db.add(np.array(document_embeddings))
        
        # Store documents for retrieval
        self.documents = documents
        
        print(f"Indexed {len(documents)} documents")
    
    def _create_match_document(self, match_data):
        """Create a document from match data"""
        # Get team names from IDs
        batting_team_id = match_data['batting_team'].iloc[0]
        bowling_team_id = match_data['bowling_team'].iloc[0]
        
        batting_team = next((k for k, v in self.data['team'].items() if v == batting_team_id), f"Team {batting_team_id}")
        bowling_team = next((k for k, v in self.data['team'].items() if v == bowling_team_id), f"Team {bowling_team_id}")
        
        # Get venue name from ID
        venue_id = match_data['venue'].iloc[0]
        venue_name = next((k for k, v in self.data['stadium'].items() if v == venue_id), f"Venue {venue_id}")
        
        date = match_data['date'].iloc[0]
        
        # Create match summary
        content = f"Match: {batting_team} vs {bowling_team} at {venue_name} on {date}\n\n"
        
        # Add batting statistics
        content += "Batting Statistics:\n"
        batters = match_data.groupby('batter')
        for batter_id, batter_data in batters:
            batter_name = next((k for k, v in self.data['player'].items() if v == batter_id), f"Player {batter_id}")
            runs = batter_data['runs_batter'].sum()
            balls = len(batter_data)
            content += f"{batter_name}: {runs} runs from {balls} balls\n"
        
        # Add bowling statistics
        content += "\nBowling Statistics:\n"
        bowlers = match_data.groupby('bowler')
        for bowler_id, bowler_data in bowlers:
            bowler_name = next((k for k, v in self.data['player'].items() if v == bowler_id), f"Player {bowler_id}")
            wickets = bowler_data['is_wicket'].sum()
            runs = bowler_data['runs_batter'].sum() + bowler_data['extras'].sum()
            overs = len(bowler_data) // 6  # Approximate overs
            content += f"{bowler_name}: {wickets} wickets for {runs} runs in {overs} overs\n"
        
        return {
            "id": f"match_{date}_{batting_team_id}_{bowling_team_id}",
            "type": "match",
            "content": content
        }
    
    def _create_player_document(self, player_name, player_id):
        """Create a document from player data"""
        content = f"Player: {player_name} (ID: {player_id})\n\n"
        
        # Filter cricket data for this player
        if self.cricket_df is not None:
            # Batting stats
            batting_data = self.cricket_df[self.cricket_df['batter'] == player_id]
            if not batting_data.empty:
                total_runs = batting_data['runs_batter'].sum()
                total_balls = len(batting_data)
                content += f"Batting Statistics:\n"
                content += f"Total Runs: {total_runs}\n"
                content += f"Total Balls Faced: {total_balls}\n"
                if total_balls > 0:
                    strike_rate = (total_runs / total_balls) * 100
                    content += f"Strike Rate: {strike_rate:.2f}\n"
            
            # Bowling stats
            bowling_data = self.cricket_df[self.cricket_df['bowler'] == player_id]
            if not bowling_data.empty:
                total_wickets = bowling_data['is_wicket'].sum()
                total_runs_conceded = bowling_data['runs_batter'].sum() + bowling_data['extras'].sum()
                content += f"\nBowling Statistics:\n"
                content += f"Total Wickets: {total_wickets}\n"
                content += f"Total Runs Conceded: {total_runs_conceded}\n"
        
        return {
            "id": f"player_{player_id}",
            "type": "player",
            "content": content
        }
    
    def _create_team_document(self, team_name, team_id):
        """Create a document from team data"""
        content = f"Team: {team_name} (ID: {team_id})\n\n"
        
        # Filter cricket data for this team
        if self.cricket_df is not None:
            # Matches as batting team
            batting_matches = self.cricket_df[self.cricket_df['batting_team'] == team_id]
            batting_match_count = len(batting_matches.groupby(['date', 'venue', 'bowling_team']))
            
            # Matches as bowling team
            bowling_matches = self.cricket_df[self.cricket_df['bowling_team'] == team_id]
            bowling_match_count = len(bowling_matches.groupby(['date', 'venue', 'batting_team']))
            
            total_matches = batting_match_count + bowling_match_count
            content += f"Total Matches: {total_matches}\n"
            
            # Add more team statistics as needed
        
        return {
            "id": f"team_{team_id}",
            "type": "team",
            "content": content
        }
    
    def retrieve_relevant_documents(self, query, top_k=3):
        """Retrieve relevant documents for a query"""
        if self.vector_db is None or self.embeddings_model is None:
            print("Retrieval system not set up. Call setup_retrieval_system() first.")
            return []
        
        # Encode the query
        query_embedding = self.embeddings_model.encode(query)
        query_embedding = np.array([query_embedding])
        
        # Search for similar documents
        distances, indices = self.vector_db.search(query_embedding, top_k)
        
        # Get the documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        
        return retrieved_docs
    
    def generate_response_with_retrieval(self, instruction, input_text="", max_length=512):
        """Generate a response using retrieval augmentation"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(instruction + " " + input_text)
        
        # Create context from retrieved documents
        context = "I'll answer based on the following information:\n\n"
        for doc in retrieved_docs:
            context += doc['content'] + "\n\n"
        
        # Combine context with the original query
        augmented_input = context + "\nNow, answering the original question: " + input_text
        
        # Generate response with the augmented input
        return self.generate_response(instruction, augmented_input, max_length)
    
    def interactive_mode(self, use_retrieval=True):
        """Run an interactive session with the model"""
        print("Starting interactive mode. Type 'exit' to quit.")
        print(f"Retrieval augmentation: {'Enabled' if use_retrieval else 'Disabled'}")
        
        while True:
            instruction = input("\nUser: ")
            if instruction.lower() == 'exit':
                break
            
            if use_retrieval:
                response = self.generate_response_with_retrieval(instruction)
            else:
                response = self.generate_response(instruction)
                
            print(f"\nAssistant: {response}")

    # You can add this method if you want to generate synthetic examples
    def _generate_synthetic_examples(self):
        """Generate synthetic examples for training"""
        print("Generating synthetic examples...")
        
        # This is a placeholder implementation
        # You can customize this to generate whatever synthetic examples you need
        examples = []
        
        # Example: Generate simple questions about cricket rules
        examples.append({
            "instruction": "How many players are in a cricket team?",
            "input": "",
            "output": "A cricket team consists of 11 players on the field."
        })
        
        examples.append({
            "instruction": "What is LBW in cricket?",
            "input": "",
            "output": "LBW stands for 'Leg Before Wicket'. It's a way a batter can be dismissed when the ball hits their body (usually the leg) when it would otherwise have hit the wicket."
        })
        
        # Add more synthetic examples as needed
        
        print(f"Generated {len(examples)} synthetic examples")
        return examples

    def prepare_dataset(self, training_data):
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        # Ensure all examples have the same structure
        cleaned_data = []
        for item in training_data:
            if "instruction" in item and "input" in item and "output" in item:
                cleaned_data.append(item)
        
        # Create dataset
        dataset = Dataset.from_list(cleaned_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Format the prompt consistently
            prompts = []
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
                else:
                    prompt = f"Instruction: {instruction}\nOutput: "
                prompts.append(prompt)
            
            # Tokenize inputs
            tokenized_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=512,  # Adjust this based on your model's context window
                return_tensors="pt"
            )
            
            # Tokenize outputs (labels)
            tokenized_outputs = self.tokenizer(
                examples["output"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Adjust this based on your model's context window
                return_tensors="pt"
            )
            
            # Create labels with -100 for non-output tokens
            labels = tokenized_inputs["input_ids"].clone()
            for i in range(len(labels)):
                # Find the position where the output starts
                output_start = len(self.tokenizer.encode(prompts[i], add_special_tokens=False))
                # Set non-output tokens to -100
                labels[i, :output_start] = -100
            
            return {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": labels
            }
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
        
        return tokenized_dataset

def main():
    # Initialize the model
    cricket_model = CricketAnalysisModel()
    
    # Extract zip data if needed
    if os.path.exists("cleaned_data.zip"):
        cricket_model.extract_zip_data()
    
    # Load data
    cricket_model.load_data()
    
    # Set up retrieval system
    cricket_model.setup_retrieval_system()
    
    # Initialize model
    cricket_model.initialize_model()
    
    # Train the model (optional - you can skip this if you just want to use retrieval)
    # cricket_model.train()
    
    # Start interactive mode with retrieval
    cricket_model.interactive_mode(use_retrieval=True)

if __name__ == "__main__":
    main()

# Get a player ID
player_id = cricket_mappings.get_player_id("Virat Kohli")

# Get a team name
team_name = cricket_mappings.get_team_name(24)  # India

# Look up a stadium
stadium_name = cricket_mappings.get_stadium_name(5)  # Melbourne Cricket Ground
