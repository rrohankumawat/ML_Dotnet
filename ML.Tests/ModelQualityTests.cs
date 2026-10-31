using Microsoft.ML;
using ML_Dotnet;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Tests
{
    public class ModelQualityTests
    {
        [Fact]
        public void Model_reaches_minimum_accuracy()
        {
            var mlContext = new MLContext(seed: 0);

            var (_, metrics) = TicketTrainer.TrainAndEvaluate(mlContext);

            Assert.True(metrics.MicroAccuracy >= 0.80,$"Accuracy was only {metrics.MicroAccuracy:P1}");
        }
    }
}
