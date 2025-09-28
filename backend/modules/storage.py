# modules/storage.py
import json
import logging
import pandas as pd
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


def save_to_json(data: List[Dict], filename: str):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Merged comments saved to JSON: {filename}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")


def save_to_csv(data: List[Dict], filename: str):
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"Merged comments saved to CSV: {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")


def generate_ai_summary(result: Dict, content_id: str, api_key: str) -> str:
    logger.info("Preparing prompt for AI summary...")
    insight = result.get('insight')
    if not insight:
        return "Insight data not found."

    prompt = (
        f"Anda adalah seorang analis media sosial. Buat ringkasan laporan deskriptif untuk konten video '{content_id}' "
        "berdasarkan hasil data yang didapat berupa banyaknya sentimen dari konten video, topic apa saja yang dibicarakan "
        "pada kolom komentar. dan buatlah hipotesa dari hasil tersebut. Misal jika konten tersebut banyak sentimen negatif "
        "berarti sentimen negatif apa saja yang dibahas berdasarkan topik komentar, begitu juga dengan positif dan netral.\n\n"
        f"Ringkasan Umum: {insight.summary}\n\nTopik Utama yang Dibahas:\n"
    )

    for topic in insight.topic_details:
        if topic.get('topic') != 'Lainnya':
            prompt += (
                f"- Topik '{topic.get('topic')}' ({topic.get('percentage'):.1f}%) "
                f"dengan kata kunci: {', '.join(topic.get('keywords', []))}.\n"
            )

    logger.info("Contacting Qwen API for summary...")
    try:
        client = OpenAI(
            api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        completion = client.chat.completions.create(
            model="qwen-flash", messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        summary = completion.choices[0].message.content
        logger.info("AI summary generated successfully.")
        return summary
    except Exception as e:
        logger.error(f"Failed to contact Qwen API: {e}")
        return "Failed to generate AI summary due to an API error."


def save_summary_to_txt(summary_text: str, filename: str):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f"AI summary report saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save summary file: {e}")


def save_insight_summary(insight_summary: str, filename: str):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(insight_summary)
        logger.info(f"Insight summary saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save insight summary file: {e}")
