import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionLlama3Explain(Action):
    def name(self) -> Text:
        return "action_llama3_explain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        if not user_message:
            dispatcher.utter_message(text="Can you please repeat your question?")
            return []   

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",  
                    "prompt": user_message,
                    "stream": False
                }
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "").strip()
                if answer:
                    dispatcher.utter_message(text=answer)
                else:
                    dispatcher.utter_message(text="I couldn't find an explanation for that.")
            else:
                print(f"❌ Ollama API Error - Status Code: {response.status_code}")
                dispatcher.utter_message(text="Sorry, I had trouble generating the answer.")

        except Exception as e:
            print(f"🔥 Error in Ollama request: {e}")
            dispatcher.utter_message(text="Sorry, something went wrong while contacting the model.")

        return []

