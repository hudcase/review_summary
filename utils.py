
def apply_chat_template(system, user):

    prompt =  f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt

def generate_review_title_prompt_tmp0(review_body):
    system_prompt = (
        "你将看到一个来自电商平台的商品评论, 请根据评论内容，生成简短总结。"
    )
    user_prompt = review_body

    # 使用 apply_chat_template 格式
    return apply_chat_template(system_prompt, user_prompt)



def generate_review_title_prompt_tmp1(review_body):
    system_prompt = (
        "你是一个电商平台商品评论摘要生成器。你将根据用户的评论正文（review_body）生成一个简短的评论标题（review_title）。"
    )
    user_prompt = review_body

    # 使用 apply_chat_template 格式
    return apply_chat_template(system_prompt, user_prompt)
