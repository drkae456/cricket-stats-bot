"""
Module to load and provide access to all cricket data mappings.
"""
import json
import os
from typing import Dict, Any

class CricketMappings:
    """
    Class to load and provide access to all cricket data mappings.
    """
    def __init__(self, data_dir: str = "cleaned_data"):
        """
        Initialize the mappings by loading all mapping files.
        
        Args:
            data_dir: Directory containing the mapping files
        """
        self.data_dir = data_dir
        
        # Load all mappings
        self.player_mapping = self._load_mapping("player_mapping.json")
        self.team_mapping = self._load_mapping("team_mapping.json")
        self.stadium_mapping = self._load_mapping("stadium_mapping.json")
        self.tournament_mapping = self._load_mapping("tournament_mapping.json")
        self.format_mapping = self._load_mapping("format_mapping.json")
        self.dismissal_mapping = self._load_mapping("dismissal_mapping.json")
        self.extras_type_mapping = self._load_mapping("extras_type_mapping.json")
        
        # Create reverse mappings for lookups by ID
        self.player_mapping_reverse = self._create_reverse_mapping(self.player_mapping)
        self.team_mapping_reverse = self._create_reverse_mapping(self.team_mapping)
        self.stadium_mapping_reverse = self._create_reverse_mapping(self.stadium_mapping)
        self.tournament_mapping_reverse = self._create_reverse_mapping(self.tournament_mapping)
        self.format_mapping_reverse = self._create_reverse_mapping(self.format_mapping)
        self.dismissal_mapping_reverse = self._create_reverse_mapping(self.dismissal_mapping)
        self.extras_type_mapping_reverse = self._create_reverse_mapping(self.extras_type_mapping)
    
    def _load_mapping(self, filename: str) -> Dict[str, int]:
        """
        Load a mapping file from the data directory.
        
        Args:
            filename: Name of the mapping file to load
            
        Returns:
            Dictionary containing the mapping
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Mapping file {filepath} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {filepath}")
            return {}
    
    def _create_reverse_mapping(self, mapping: Dict[str, int]) -> Dict[int, str]:
        """
        Create a reverse mapping (id -> name) from a name -> id mapping.
        
        Args:
            mapping: Dictionary with name -> id mapping
            
        Returns:
            Dictionary with id -> name mapping
        """
        return {v: k for k, v in mapping.items()}
    
    def get_player_id(self, player_name: str) -> int:
        """Get player ID from player name"""
        return self.player_mapping.get(player_name)
    
    def get_player_name(self, player_id: int) -> str:
        """Get player name from player ID"""
        return self.player_mapping_reverse.get(player_id)
    
    def get_team_id(self, team_name: str) -> int:
        """Get team ID from team name"""
        return self.team_mapping.get(team_name)
    
    def get_team_name(self, team_id: int) -> str:
        """Get team name from team ID"""
        return self.team_mapping_reverse.get(team_id)
    
    def get_stadium_id(self, stadium_name: str) -> int:
        """Get stadium ID from stadium name"""
        return self.stadium_mapping.get(stadium_name)
    
    def get_stadium_name(self, stadium_id: int) -> str:
        """Get stadium name from stadium ID"""
        return self.stadium_mapping_reverse.get(stadium_id)
    
    def get_tournament_id(self, tournament_name: str) -> int:
        """Get tournament ID from tournament name"""
        return self.tournament_mapping.get(tournament_name)
    
    def get_tournament_name(self, tournament_id: int) -> str:
        """Get tournament name from tournament ID"""
        return self.tournament_mapping_reverse.get(tournament_id)
    
    def get_format_id(self, format_name: str) -> int:
        """Get format ID from format name"""
        return self.format_mapping.get(format_name)
    
    def get_format_name(self, format_id: int) -> str:
        """Get format name from format ID"""
        return self.format_mapping_reverse.get(format_id)
    
    def get_dismissal_id(self, dismissal_type: str) -> int:
        """Get dismissal ID from dismissal type"""
        return self.dismissal_mapping.get(dismissal_type)
    
    def get_dismissal_type(self, dismissal_id: int) -> str:
        """Get dismissal type from dismissal ID"""
        return self.dismissal_mapping_reverse.get(dismissal_id)
    
    def get_extras_type_id(self, extras_type: str) -> int:
        """Get extras type ID from extras type"""
        return self.extras_type_mapping.get(extras_type)
    
    def get_extras_type(self, extras_type_id: int) -> str:
        """Get extras type from extras type ID"""
        return self.extras_type_mapping_reverse.get(extras_type_id)


# Create a singleton instance for easy import
cricket_mappings = CricketMappings()

# Example usage:
if __name__ == "__main__":
    # Test the mappings
    print(f"Test format: {cricket_mappings.get_format_name(1)} -> {cricket_mappings.get_format_id('Test')}")
    print(f"ODI format: {cricket_mappings.get_format_name(2)} -> {cricket_mappings.get_format_id('ODI')}")
    
    # Test player lookup
    player_name = "Virat Kohli"  # Replace with a player in your mapping
    player_id = cricket_mappings.get_player_id(player_name)
    if player_id:
        print(f"Player: {player_name} -> ID: {player_id} -> Name: {cricket_mappings.get_player_name(player_id)}")
    
    # Test team lookup
    team_name = "India"  # Replace with a team in your mapping
    team_id = cricket_mappings.get_team_id(team_name)
    if team_id:
        print(f"Team: {team_name} -> ID: {team_id} -> Name: {cricket_mappings.get_team_name(team_id)}") 