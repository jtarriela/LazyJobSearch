"""
YAML Writer Service

This module handles writing company seed YAML files to the user's
configuration directory with proper validation and organization.
"""
from __future__ import annotations
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .models import CompanySeed

logger = logging.getLogger(__name__)


class YamlWriterService:
    """Service for writing and managing company seed YAML files"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize YAML writer service
        
        Args:
            config_dir: Directory to store company seeds (defaults to ~/.lazyjobsearch)
        """
        if config_dir is None:
            config_dir = Path.home() / '.lazyjobsearch'
        
        self.config_dir = config_dir
        self.companies_dir = config_dir / 'companies'
        self.index_file = config_dir / 'companies_index.yaml'
        
        # Ensure directories exist
        self.companies_dir.mkdir(parents=True, exist_ok=True)
    
    def write_company_seed(self, seed: CompanySeed, overwrite: bool = False) -> Path:
        """
        Write company seed to YAML file
        
        Args:
            seed: CompanySeed instance to write
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to written file
            
        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        file_path = self.companies_dir / f"{seed.id}.yaml"
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Company seed file already exists: {file_path}")
        
        logger.info(f"Writing company seed for {seed.name} to {file_path}")
        
        # Convert to dict and clean up for YAML output
        seed_dict = self._prepare_for_yaml(seed)
        
        # Write YAML file
        with file_path.open('w', encoding='utf-8') as f:
            yaml.dump(seed_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Update index
        self._update_index(seed)
        
        logger.info(f"Successfully wrote company seed: {file_path}")
        return file_path
    
    def read_company_seed(self, company_id: str) -> Optional[CompanySeed]:
        """
        Read company seed from YAML file
        
        Args:
            company_id: Company slug ID
            
        Returns:
            CompanySeed instance or None if not found
        """
        file_path = self.companies_dir / f"{company_id}.yaml"
        
        if not file_path.exists():
            logger.warning(f"Company seed file not found: {file_path}")
            return None
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return CompanySeed(**data)
            
        except Exception as e:
            logger.error(f"Error reading company seed {file_path}: {e}")
            return None
    
    def list_company_seeds(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available company seeds with metadata
        
        Returns:
            Dictionary mapping company_id to metadata
        """
        try:
            if not self.index_file.exists():
                return {}
            
            with self.index_file.open('r', encoding='utf-8') as f:
                index = yaml.safe_load(f) or {}
            
            return index
            
        except Exception as e:
            logger.error(f"Error reading company index: {e}")
            return {}
    
    def delete_company_seed(self, company_id: str) -> bool:
        """
        Delete company seed file and remove from index
        
        Args:
            company_id: Company slug ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        file_path = self.companies_dir / f"{company_id}.yaml"
        
        if not file_path.exists():
            logger.warning(f"Company seed file not found for deletion: {file_path}")
            return False
        
        try:
            # Remove file
            file_path.unlink()
            
            # Remove from index
            self._remove_from_index(company_id)
            
            logger.info(f"Successfully deleted company seed: {company_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting company seed {company_id}: {e}")
            return False
    
    def _prepare_for_yaml(self, seed: CompanySeed) -> Dict[str, Any]:
        """Prepare CompanySeed for YAML serialization"""
        # Convert to dict using Pydantic
        seed_dict = seed.model_dump()
        
        # Convert HttpUrl objects to strings
        if 'careers' in seed_dict:
            if 'primary_url' in seed_dict['careers']:
                seed_dict['careers']['primary_url'] = str(seed_dict['careers']['primary_url'])
            if 'discovered_alternatives' in seed_dict['careers']:
                seed_dict['careers']['discovered_alternatives'] = [
                    str(url) for url in seed_dict['careers']['discovered_alternatives']
                ]
        
        if 'crawler' in seed_dict and 'start_urls' in seed_dict['crawler']:
            seed_dict['crawler']['start_urls'] = [
                str(url) for url in seed_dict['crawler']['start_urls']
            ]
        
        return seed_dict
    
    def _update_index(self, seed: CompanySeed) -> None:
        """Update the company index file"""
        try:
            # Load existing index
            index = {}
            if self.index_file.exists():
                with self.index_file.open('r', encoding='utf-8') as f:
                    index = yaml.safe_load(f) or {}
            
            # Update index entry
            index[seed.id] = {
                'name': seed.name,
                'domain': seed.domain,
                'portal_type': seed.portal.type.value,
                'careers_url': str(seed.careers.primary_url),
                'created_at': seed.metadata.get('created_at'),
                'updated_at': datetime.utcnow().isoformat(),
                'notes': seed.notes,
            }
            
            # Write updated index
            with self.index_file.open('w', encoding='utf-8') as f:
                yaml.dump(index, f, default_flow_style=False, sort_keys=True, allow_unicode=True)
                
        except Exception as e:
            logger.error(f"Error updating company index: {e}")
    
    def _remove_from_index(self, company_id: str) -> None:
        """Remove company from index file"""
        try:
            if not self.index_file.exists():
                return
            
            # Load existing index
            with self.index_file.open('r', encoding='utf-8') as f:
                index = yaml.safe_load(f) or {}
            
            # Remove entry
            index.pop(company_id, None)
            
            # Write updated index
            with self.index_file.open('w', encoding='utf-8') as f:
                yaml.dump(index, f, default_flow_style=False, sort_keys=True, allow_unicode=True)
                
        except Exception as e:
            logger.error(f"Error removing from company index: {e}")
    
    def generate_dry_run_yaml(self, seed: CompanySeed) -> str:
        """
        Generate YAML content for dry-run preview
        
        Args:
            seed: CompanySeed instance
            
        Returns:
            YAML content as string
        """
        seed_dict = self._prepare_for_yaml(seed)
        return yaml.dump(seed_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)