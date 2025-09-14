# config/logging_config.py
import logging
import logging.config
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class LLMPerformanceLogger:
    """Logger for tracking LLM performance metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Performance metrics file
        self.perf_file = self.log_dir / "llm_performance.jsonl"
        
        # Setup logger
        self.logger = logging.getLogger("llm_performance")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / "llm_performance.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_generation(
        self, 
        model_name: str, 
        prompt_length: int, 
        response_length: int, 
        generation_time: float, 
        success: bool, 
        error_message: Optional[str] = None,
        model_type: str = "unknown",
        is_fallback: bool = False
    ):
        """Log a generation event"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_type": model_type,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "generation_time": generation_time,
            "success": success,
            "error_message": error_message,
            "is_fallback": is_fallback,
            "tokens_per_second": response_length / generation_time if generation_time > 0 else 0
        }
        
        # Log to JSONL file for detailed analysis
        with open(self.perf_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Log to standard logger
        status = "SUCCESS" if success else "FAILED"
        fallback_info = " (FALLBACK)" if is_fallback else ""
        
        self.logger.info(
            f"{status}{fallback_info} - {model_name} - "
            f"Time: {generation_time:.2f}s - "
            f"Tokens/s: {metrics['tokens_per_second']:.1f}"
        )
        
        if error_message:
            self.logger.error(f"Error in {model_name}: {error_message}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        
        if not self.perf_file.exists():
            return {}
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_metrics = []
        
        try:
            with open(self.perf_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics = json.loads(line)
                        event_time = datetime.fromisoformat(metrics['timestamp']).timestamp()
                        if event_time >= cutoff_time:
                            recent_metrics.append(metrics)
        except Exception as e:
            self.logger.error(f"Error reading performance metrics: {e}")
            return {}
        
        if not recent_metrics:
            return {"message": "No recent metrics found"}
        
        # Calculate summary statistics
        summary = {}
        
        # Group by model
        by_model = {}
        for metric in recent_metrics:
            model = metric['model_name']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(metric)
        
        for model, metrics in by_model.items():
            successful = [m for m in metrics if m['success']]
            failed = [m for m in metrics if not m['success']]
            fallback_uses = [m for m in metrics if m['is_fallback']]
            
            summary[model] = {
                "total_requests": len(metrics),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "fallback_uses": len(fallback_uses),
                "success_rate": len(successful) / len(metrics) if metrics else 0,
                "avg_generation_time": sum(m['generation_time'] for m in successful) / len(successful) if successful else 0,
                "avg_tokens_per_second": sum(m['tokens_per_second'] for m in successful) / len(successful) if successful else 0,
                "total_tokens_generated": sum(m['response_length'] for m in successful)
            }
        
        summary['_metadata'] = {
            "hours_analyzed": hours,
            "total_events": len(recent_metrics),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return summary


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup comprehensive logging configuration"""
    
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            }
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard"
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": str(log_dir / "hybrid_llm.log"),
                "formatter": "detailed"
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.FileHandler",
                "filename": str(log_dir / "errors.log"),
                "formatter": "detailed"
            }
        },
        "loggers": {
            "gemini_llm": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "langchain_agent": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "api_keys": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "llm_performance": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file", "error_file"]
        }
    }
    
    logging.config.dictConfig(config)
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("=== Hybrid LLM System Started ===")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger


def log_system_info():
    """Log system information for debugging"""
    logger = logging.getLogger(__name__)
    
    import torch
    import sys
    import platform
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Log environment variables (safely)
    env_vars = ['GEMINI_API_KEY', 'HF_TOKEN', 'OPENAI_API_KEY']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"{var}: {'*' * min(len(value), 10)}... (configured)")
        else:
            logger.info(f"{var}: Not set")


class ModelSwitchLogger:
    """Logger for tracking model switches and decisions"""
    
    def __init__(self, log_dir: str = "logs"):
        self.logger = logging.getLogger("model_switch")
        self.switch_file = Path(log_dir) / "model_switches.jsonl"
        
    def log_switch(
        self, 
        from_model: str, 
        to_model: str, 
        reason: str, 
        user_initiated: bool = False
    ):
        """Log a model switch event"""
        
        switch_event = {
            "timestamp": datetime.now().isoformat(),
            "from_model": from_model,
            "to_model": to_model,
            "reason": reason,
            "user_initiated": user_initiated
        }
        
        # Log to JSONL file
        with open(self.switch_file, 'a') as f:
            f.write(json.dumps(switch_event) + '\n')
        
        # Log to standard logger
        switch_type = "USER" if user_initiated else "AUTO"
        self.logger.info(f"{switch_type} SWITCH: {from_model} → {to_model} - {reason}")
    
    def log_fallback(
        self, 
        primary_model: str, 
        fallback_model: str, 
        error_message: str
    ):
        """Log a fallback event"""
        self.log_switch(
            from_model=primary_model,
            to_model=fallback_model,
            reason=f"Primary model failed: {error_message}",
            user_initiated=False
        )


# Initialize global performance logger
_performance_logger = None

def get_performance_logger() -> LLMPerformanceLogger:
    """Get global performance logger instance"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = LLMPerformanceLogger()
    return _performance_logger


def get_model_switch_logger() -> ModelSwitchLogger:
    """Get global model switch logger instance"""
    return ModelSwitchLogger()


if __name__ == "__main__":
    # Test logging setup
    setup_logging("DEBUG")
    log_system_info()
    
    # Test performance logger
    perf_logger = get_performance_logger()
    perf_logger.log_generation(
        model_name="test_model",
        prompt_length=100,
        response_length=200,
        generation_time=1.5,
        success=True,
        model_type="local"
    )
    
    print("Logging setup completed successfully!")
