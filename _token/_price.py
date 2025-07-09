from transformers import AutoTokenizer

# 计算token
def tokens(question:str, answer:str) -> int:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer)
    total_tokens = len(question_tokens) + len(answer_tokens)
    return total_tokens

# 价格计算
def price(tokens:int):
    words = 1000
    price = 0.2
    total_price = (tokens / words) * price
    return total_price

# 计算推理价格
def agent_tokens_price(question:str, answer:str):
    total_tokens = tokens(question, answer)
    total_price = price(total_tokens)
    return {"price":total_price, "tokens":total_tokens}

# 计算缓存价格
def cache_tokens_price(question:str, answer:str):
    cache_price = 0.1
    total_tokens = tokens(question, answer)
    total_price = cache_price * price(total_tokens)
    return {"price":total_price, "tokens":total_tokens}

if __name__ == "__main__":
    # res1 = cache_tokens_price("你好", "你好啊")
    # print("缓存价格：",res1)
    # res2 = agent_tokens_price("你好", "你好啊")
    # print("代理价格：",res2)
    pass
