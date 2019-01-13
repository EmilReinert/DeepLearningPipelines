import pipelines.defect_prediction as dp

if __name__ == "__main__":
    """
    Main to test-run specific pipelines or modules
    """
    dp_pipeline = dp.DefectPrediction('/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/data/defective_test.py',
                                 '/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/data/clean_test.py',
                                 '/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/data/testdata_test.py')
    dp_pipeline.run()
    print("run completed")
