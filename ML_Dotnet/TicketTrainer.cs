using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Dotnet
{
    public static class TicketTrainer
    {
        public static (ITransformer Model,
            MulticlassClassificationMetrics Metrics)
            TrainAndEvaluate(MLContext mlContext)
        {
            string filepath = AppContext.BaseDirectory + "tickets.tsv";

            var data = mlContext.Data.LoadFromTextFile<Ticket>(filepath, new TextLoader.Options
            {
                HasHeader = false,
                Separators = new[] { '\t' }
            });

            var splits = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 2);


            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(Ticket.Category)).
                            Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(Ticket.Text))).Append(
                            mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")).Append(
                            mlContext.Transforms.Conversion.MapKeyToValue("PredictedCategory", "PredictedLabel"));

            var model = pipeline.Fit(splits.TrainSet);

            var prediction = model.Transform(splits.TestSet);

            var metrics = mlContext.MulticlassClassification.Evaluate(prediction, labelColumnName: "Label");


            return (model, metrics);
        }
    }
}
