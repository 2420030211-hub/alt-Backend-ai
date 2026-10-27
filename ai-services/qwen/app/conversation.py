from typing import List, Dict
import time

class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(conversation_id, [])

    def append_turn(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = []
        self.sessions[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        # Keep recent 20 turns to protect context window
        if len(self.sessions[conversation_id]) > 20:
            self.sessions[conversation_id] = self.sessions[conversation_id][-20:]

conversation_mgr = ConversationManager()
