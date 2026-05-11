from collections import defaultdict
from typing import Dict, Any, List, Callable, Optional

class ReplayBuffer:
    def __init__(self):
        self.episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.state_map: Dict[str, Dict[str, Any]] = {}

    def store(self, state: Dict[str, Any], action: Any = None):
        uuid = state.get("uuid")
        state_id = state.get("stateId")
        parent_id = state.get("parentStateId")
        done = (state.get("status") == "finished")
        
        transition = {
            "current_state": state,
            "action": action, 
            "reward": None, 
            "next_state": None, 
            "done": done
        }
        
        if parent_id and parent_id != "Null" and parent_id in self.state_map:
            self.state_map[parent_id]["next_state"] = state
            
        self.episodes[uuid].append(transition)
        self.state_map[state_id] = transition

    def update_reward(self, state_id: str, reward: float):
        if state_id in self.state_map:
            self.state_map[state_id]["reward"] = reward

    def get_episode(self, uuid: str) -> List[Dict[str, Any]]:
        return self.episodes.get(uuid, [])

    def remove_episode(self, uuid: str):
        if uuid in self.episodes:
            # Clean up state_map to prevent memory leaks
            for transition in self.episodes[uuid]:
                state_id = transition["current_state"].get("stateId")
                if state_id in self.state_map:
                    del self.state_map[state_id]
            del self.episodes[uuid]

    def __len__(self):
        return sum(len(ep) for ep in self.episodes.values())