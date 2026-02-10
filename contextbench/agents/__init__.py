"""Unified agent trajectory extraction interface."""

from .minisweagent import extract_trajectory as extract_miniswe
from .sweagent import extract_trajectory as extract_swe
from .agentless import extract_trajectory as extract_agentless
from .prometheus import extract_trajectory as extract_prometheus
from .openhands import extract_trajectory as extract_openhands


def extract_trajectory(traj_file_or_data) -> dict:
    """Auto-detect format and extract trajectory.
    
    Supports:
    - MiniSWE-agent: .traj.json files
    - SWE-agent: .checkpoints.jsonl files
    - Agentless: *_traj.json files
    - Prometheus: .log files (Prometheus answer_issue_logs format)
    - OpenHands: output.jsonl files or dict data with 'history' field
    
    Args:
        traj_file_or_data: Either a file path (str) or pre-parsed OpenHands data (dict)
    
    Returns unified format:
    {
        'pred_steps': [{'files': [...], 'spans': {...}}, ...],
        'pred_files': [...],
        'pred_spans': {...}
    }
    """
    # Handle dict input (OpenHands pre-parsed data)
    if isinstance(traj_file_or_data, dict):
        if 'history' in traj_file_or_data:
            return extract_openhands(traj_file_or_data)
        else:
            raise ValueError(f"Unsupported dict format (no 'history' field)")
    
    # Handle file path input
    traj_file = traj_file_or_data
    if (traj_file.endswith('.checkpoints.jsonl') 
        or traj_file.endswith('.context.json') 
        or traj_file.endswith('patch_context.txt')
        or traj_file.endswith('.traj')):
        return extract_swe(traj_file)
    elif traj_file.endswith('.traj.json'):
        return extract_miniswe(traj_file)
    elif traj_file.endswith('_traj.json'):
        return extract_agentless(traj_file)
    elif traj_file.endswith('output.jsonl'):
        return extract_openhands(traj_file)
    elif traj_file.endswith('.log'):
        # Prometheus .log files can be very large and the context markers may not
        # appear in the first few KB. Let the Prometheus extractor decide.
        data = extract_prometheus(traj_file)
        if data.get("pred_steps") or data.get("pred_files") or data.get("pred_spans"):
            return data
        raise ValueError(f"Unsupported .log trajectory format: {traj_file}")
    else:
        raise ValueError(f"Unsupported trajectory format: {traj_file}")


__all__ = ['extract_trajectory']
