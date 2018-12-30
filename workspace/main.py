import pipelines.defectPrediction_deepTreeLstm as dpLSTM

if __name__ == "__main__":
    """
    Main to test-run specific pipelines or modules
    """
    dp = dpLSTM.DefectPredictor('/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/data/defective_test.py',
                            '/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/data/clean_test.py')
    dp.run()
    print("hi")
