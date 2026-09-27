"""Optional conversation adapter for local audio testing; no robot or harness tools."""


class OpenAIChat:
    def __init__(self, key, model='gpt-5-mini'):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=key, timeout=40, max_retries=0)
        self.model, self.history = model, []

    async def respond(self, text):
        message = {'role': 'user', 'content': text}
        result = await self.client.responses.create(
            model=self.model, store=False, max_output_tokens=1024,
            reasoning={'effort': 'minimal'}, text={'verbosity': 'low'},
            instructions=(
                'You are Reins, a friendly robot voice assistant in a local conversation test. '
                f'Your configured language model is {self.model}. If asked which model you use, '
                'report that exact configured name; do not guess a different model. '
                'Answer in the user\'s language in one or two short sentences, at most 500 characters. '
                'Use plain spoken text without markdown. You have no camera, robot control, '
                'motion tools or access to the simulator. Never claim to have moved a robot '
                'or observed the room. Explain that this is a conversation test if asked to act.'),
            input=self.history + [message])
        answer = result.output_text.strip()
        if result.status != 'completed' or not 1 <= len(answer) <= 1000:
            raise ValueError('The conversation model returned no complete, bounded answer')
        self.history = (self.history + [message, {'role': 'assistant', 'content': answer}])[-12:]
        return answer

    def remember_spoken(self, text):
        self.history = (self.history + [{'role': 'assistant', 'content': text}])[-12:]

    async def close(self):
        self.history.clear()
        await self.client.close()
