import asyncio
from tqdm.asyncio import tqdm_asyncio
import sys

class LLMOpenAIClient:
    def __init__(self,
                 client,
                 concurrency=100,
                 progress_callback=None):
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.progress_callback = progress_callback

    async def calling_llm(self,
                          messages: list,
                          model: str,
                          temperature: float,
                          top_p: float,
                          max_tries: int = 3,
                          backoff: float = 2.0):
        attempt = 1
        while True:
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p
                    )
                return completion.choices[0].message.content

            except Exception as e:
                print(f"Attempt {attempt} failed with error {e}", file=sys.stderr)
                if attempt >= max_tries:
                    raise
                sleep_time = backoff * attempt
                print(f"Retrying in {sleep_time} seconds...", file=sys.stderr)
                await asyncio.sleep(sleep_time)
                attempt += 1

    async def prompting_process(self,
                                messages_list: list,
                                model: str = "gpt-4o-mini",
                                temperature: float = 1.0,
                                top_p: float = 1.0,
                                max_tries: int = 3,
                                backoff: float = 2.0):
        if self.progress_callback:
            total = len(messages_list)
            completed = 0
            results = [None]*total

            async def wrapped_calling_llm(index, messages):
                nonlocal completed
                result = await self.calling_llm(messages, model, temperature, top_p, max_tries, backoff)
                completed += 1
                await self.progress_callback(completed, total)
                return index, result
            
            tasks = [asyncio.create_task(wrapped_calling_llm(i, m)) for i, m in enumerate(messages_list)]
            indexed_results = await asyncio.gather(*tasks)
            
            for index, result in indexed_results:
                results[index] = result
            
            return results

        else:
            tasks = [asyncio.create_task(self.calling_llm(m, model, temperature, top_p, max_tries, backoff))
                        for m in messages_list]
            
            results = await tqdm_asyncio.gather(*tasks)

            return results