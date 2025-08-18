"""
トークン計算とテキスト分割のユーティリティ関数
"""

import logging
import re
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """
    指定されたモデルに対応するエンコーディングを取得

    Args:
        model (str): OpenAIモデル名

    Returns:
        tiktoken.Encoding: トークナイザーエンコーディング
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # 未知のモデルの場合はデフォルトエンコーディングを使用
        logger.warning(f"Unknown model '{model}', using default o200k_base encoding")
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    テキストのトークン数をカウント

    Args:
        text (str): カウント対象のテキスト
        model (str): 使用するモデル名

    Returns:
        int: トークン数
    """
    if not text:
        return 0

    encoding = get_encoding_for_model(model)
    return len(encoding.encode(text))


def count_tokens_for_messages(
    messages: list[dict[str, Any]], model: str = "gpt-4o"
) -> int:
    """
    チャットメッセージ形式のトークン数をカウント

    Args:
        messages (list[dict]): チャットメッセージのリスト
        model (str): 使用するモデル名

    Returns:
        int: トークン数
    """
    try:
        encoding = get_encoding_for_model(model)
    except KeyError:
        logger.warning(f"Warning: model {model} not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model in {"gpt-3.5-turbo-0301"}:
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.info(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_tokens_for_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.info(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_tokens_for_messages(messages, model="gpt-4-0613")
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def split_text_by_tokens(
    text: str, max_tokens: int, model: str = "gpt-4o", overlap_tokens: int = 100
) -> list[str]:
    """
    テキストをトークン数制限に基づいて分割

    Args:
        text (str): 分割対象のテキスト
        max_tokens (int): 1チャンクあたりの最大トークン数
        model (str): 使用するモデル名
        overlap_tokens (int): チャンク間のオーバーラップトークン数

    Returns:
        list[str]: 分割されたテキストのリスト
    """
    if not text:
        return []

    # 全体のトークン数をチェック
    total_tokens = count_tokens(text, model)
    if total_tokens <= max_tokens:
        logger.debug(
            f"Text fits in single chunk: {total_tokens} tokens <= {max_tokens}"
        )
        return [text]

    logger.info(
        f"Splitting text: {total_tokens} tokens into chunks of max {max_tokens} tokens"
    )

    # 段落単位で分割を試行
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # 段落を追加した場合のトークン数を計算
        test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
        test_tokens = count_tokens(test_chunk, model)

        if test_tokens <= max_tokens:
            # まだ制限内なので追加
            current_chunk = test_chunk
        else:
            # 単一段落が制限を超える場合、文単位で分割
            paragraph_chunks = _split_paragraph_by_sentences(
                paragraph, max_tokens, model, overlap_tokens
            )
            chunks.extend(paragraph_chunks[:-1])  # 最後以外を追加
            # 最後のチャンクもトークン数をチェック
            if paragraph_chunks:
                last_chunk = paragraph_chunks[-1]
                if count_tokens(last_chunk, model) <= max_tokens:
                    current_chunk = last_chunk
                else:
                    # 最後のチャンクも制限を超える場合は強制分割
                    logger.warning(
                        "Last paragraph chunk exceeds token limit, force splitting"
                    )
                    forced_chunks = _force_split_by_chars(last_chunk, max_tokens, model)
                    chunks.extend(forced_chunks[:-1])
                    current_chunk = forced_chunks[-1] if forced_chunks else ""
            else:
                current_chunk = ""

    # Add final chunk with token validation
    if current_chunk:
        final_tokens = count_tokens(current_chunk, model)
        if final_tokens <= max_tokens:
            chunks.append(current_chunk)
        else:
            # 最終チャンクも制限を超える場合は強制分割
            logger.warning(
                f"Final chunk exceeds token limit ({final_tokens} > {max_tokens}), force splitting"
            )
            forced_chunks = _force_split_by_chars(current_chunk, max_tokens, model)
            chunks.extend(forced_chunks)
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


def _get_text_tail_by_tokens(text: str, max_tokens: int, model: str) -> str:
    """
    テキストの末尾から指定トークン数分のテキストを取得
    """
    encoding = get_encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    tail_tokens = tokens[-max_tokens:]
    return encoding.decode(tail_tokens)


def _split_paragraph_by_sentences(
    paragraph: str, max_tokens: int, model: str, overlap_tokens: int = 100
) -> list[str]:
    """
    段落を文単位で分割
    """
    # Split by sentence boundaries (Japanese and English)
    sentences = re.split(r"[.!?。！？]\s*", paragraph)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue

        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        test_tokens = count_tokens(test_chunk, model)

        if test_tokens <= max_tokens:
            current_chunk = test_chunk
        elif current_chunk:
            chunks.append(current_chunk)

            # オーバーラップ処理
            if overlap_tokens > 0:
                overlap_text = _get_text_tail_by_tokens(
                    current_chunk, overlap_tokens, model
                )
                test_chunk_with_overlap = overlap_text + " " + sentence

                # オーバーラップ後のトークン数をチェック
                if count_tokens(test_chunk_with_overlap, model) <= max_tokens:
                    current_chunk = test_chunk_with_overlap
                else:
                    # オーバーラップありでも制限を超える場合は、文のみから開始
                    current_chunk = sentence
            else:
                current_chunk = sentence
        elif count_tokens(sentence, model) > max_tokens:
            # 単一文が制限を超える場合、文字数で強制分割
            logger.warning("Single sentence exceeds token limit, force splitting")
            forced_chunks = _force_split_by_chars(sentence, max_tokens, model)
            chunks.extend(forced_chunks[:-1])
            current_chunk = forced_chunks[-1] if forced_chunks else ""
        else:
            current_chunk = sentence

    # Add final chunk with token validation
    if current_chunk:
        final_tokens = count_tokens(current_chunk, model)
        if final_tokens <= max_tokens:
            chunks.append(current_chunk)
        else:
            # 最終チャンクも制限を超える場合は強制分割
            logger.warning(
                f"Final sentence chunk exceeds token limit ({final_tokens} > {max_tokens}), force splitting"
            )
            forced_chunks = _force_split_by_chars(current_chunk, max_tokens, model)
            chunks.extend(forced_chunks)

    return chunks


def _force_split_by_chars(text: str, max_tokens: int, model: str) -> list[str]:
    """
    文字数で強制的にテキストを分割（最後の手段）
    """
    encoding = get_encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end

    return chunks


def estimate_prompt_tokens(
    user_prompt_template: str, variables: dict[str, str], model: str = "gpt-4o"
) -> int:
    """
    プロンプトテンプレートと変数から推定トークン数を計算

    Args:
        user_prompt_template (str): プロンプトテンプレート
        variables (dict): テンプレート変数
        model (str): 使用するモデル名

    Returns:
        int: 推定トークン数
    """
    # テンプレート変数を実際に置換してトークン数を計算
    try:
        filled_prompt = user_prompt_template.format(**variables)
        return count_tokens(filled_prompt, model)
    except KeyError as e:
        # 一部の変数が不足している場合は概算
        logger.warning(f"Template variable missing for token estimation: {e}")
        base_tokens = count_tokens(user_prompt_template, model)
        var_tokens = sum(count_tokens(str(v), model) for v in variables.values())
        return base_tokens + var_tokens
