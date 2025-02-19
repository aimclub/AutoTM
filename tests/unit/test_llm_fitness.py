import os

import pytest

from autotm.fitness.tm import estimate_topics_with_llm, ENV_AUTOTM_LLM_API_KEY


@pytest.mark.skipif(ENV_AUTOTM_LLM_API_KEY not in os.environ, reason=F"ChatGPT API key is not available. {ENV_AUTOTM_LLM_API_KEY} is not set.")
def test_llm_fitness_estimation(openai_api_key: str):
    topics = {
        "main0": 'entry launch remark rule satellite european build contest player test francis given author canadian cipher',
        "main1": 'engineer newspaper holland chapter staff douglas princeton tempest colostate senior executive jose nixon phoenix chemical',
        "main3": 'religion atheist belief religious atheism faith christianity existence strong universe follow theist become accept statement',
        "main4": 'population bullet arab israeli border village muslim thousand slaughter policy daughter wife authorities switzerland religious',
        "main6": 'unix directory comp package library email linux workstation graphics vendor editor user hardware export product',
        "main7": 'woman building city left child home face helmet apartment kill wife azerbaijani live father later',
        "main10": 'attack lebanese muslim hernlem israeli left troops peace fire away quite stop religion israel especially',
        "main11": 'science holland cult compass study tempest investigation methodology nixon psychology department star left colostate scientific',
        "main12": 'создавать мужчина премьер добавлять причина оставаться клуб александр сергей закон идти комитет безопасность национальный предлагать',
        "main20": 'победа россиянин чемпион американец ассоциация встречаться завершаться килограмм побеждать карьера поражение состояться всемирный категория боец одерживать поедино суметь соперник проигрывать',
        "back0": 'garbage garbage garbage',
        "back1": 'trash trash trash'
    }
    topics = {k: v.split(' ') for k, v in topics.items()}

    fitness = estimate_topics_with_llm(
        model_id="test_model",
        topics=topics,
        api_key=openai_api_key,
        max_estimated_topics=4,
        estimations_per_topic=3
    )

    print(f"Fitness: {fitness}")

    assert fitness > 0
