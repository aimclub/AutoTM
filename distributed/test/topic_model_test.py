import unittest

from kube_fitness import AVG_COHERENCE_SCORE
from kube_fitness.tm import TopicModelFactory, calculate_fitness_of_individual


class TopicModelCase(unittest.TestCase):
    def test_topic_model(self):
        topic_count = 10
        dataset_name_1 = "first"
        dataset_name_2 = "second"

        param_s = [
            66717.86348968784, 2.0, 98.42286785825902,
            80.18543570807961, 2.0, -19.948347560420373,
            -52.28141634493725, 2.0, -92.85597392137976,
            -60.49287378084627, 4.0, 3.0, 0.06840138630839943,
            0.556001061599461, 0.9894122432621849, 11679.364068753106
        ]

        metrics = [
            AVG_COHERENCE_SCORE,
            'perplexityScore', 'backgroundTokensRatioScore', 'contrast',
            'purity', 'kernelSize', 'npmi_50_list',
            'npmi_50', 'sparsity_phi', 'sparsity_theta',
            'topic_significance_uni', 'topic_significance_vacuous', 'topic_significance_back',
            'switchP_list',
            'switchP', 'all_topics',
            *(f'coherence_{i}' for i in range(10, 60, 5)),
            *(f'coherence_{i}_list' for i in range(10, 60, 5)),
        ]

        TopicModelFactory.init_factory_settings(num_processors=2, dataset_settings={
            dataset_name_1: {
                "base_path": '/home/nikolay/wspace/test_tiny_dataset',
                "topic_count": topic_count,
            },
            dataset_name_2: {
                "base_path": '/home/nikolay/wspace/test_tiny_dataset_2',
                "topic_count": topic_count,
            },
        })

        print(f"Calculating dataset {dataset_name_1}")
        fitness = calculate_fitness_of_individual(dataset_name_1, param_s, topic_count=topic_count)
        self.assertSetEqual(set(fitness.keys()), set(metrics))
        for m in metrics:
            self.assertIsNotNone(fitness[m])

        print(f"Calculating dataset {dataset_name_2}")
        fitness = calculate_fitness_of_individual(dataset_name_2, param_s, topic_count=topic_count)
        self.assertSetEqual(set(fitness.keys()), set(metrics))
        for m in metrics:
            self.assertIsNotNone(fitness[m])

        with self.assertRaises(Exception):
            calculate_fitness_of_individual("unknown_dataset", param_s, topic_count=topic_count)


if __name__ == '__main__':
    unittest.main()
