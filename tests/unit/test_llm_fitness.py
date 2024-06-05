from autotm.fitness.tm import estimate_topics_with_llm


def test_llm_fitness_estimation(openai_api_key: str):

    topics = {
        "main0": [],
        "main1": [],
        "main3": [],
        "main4": [],
        "main6": [],
        "main7": [],
        "main10": [],
        "main11": [],
        "main12": [],
        "main20": [],
        "back0": [],
        "back1": []
    }

    fitness = estimate_topics_with_llm(
        model_id="test_model",
        topics=topics,
        api_key=openai_api_key,
        max_estimated_topics=4,
        estimations_per_topic=3
    )

    assert fitness > 0
