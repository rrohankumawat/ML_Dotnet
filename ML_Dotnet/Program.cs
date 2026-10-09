
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 0);

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


Console.WriteLine(metrics.MicroAccuracy);
Console.WriteLine(metrics.MacroAccuracy);
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

var engine = mlContext.Model.CreatePredictionEngine<Ticket, TicketPrediction>(model);

var predict = engine.Predict(new Ticket { Text = "the coffee is good" });

Console.WriteLine($"Class: {predict.Category}");

var best = predict.Scores.Max();

Console.WriteLine(best);
Console.WriteLine(best > 0.75 ? $"Escalated to {predict.Category}." : "Send it to Human Agent");

public class Ticket
{
    [LoadColumn(0)]
    public string Category { get; set; } = string.Empty;
    [LoadColumn(1)]
    public string Text { get; set; } = string.Empty;
}

public class TicketPrediction
{
    [ColumnName("PredictedCategory")]
    public string Category { get; set; } = "";
    [ColumnName("Score")]
    public float[] Scores { get; set; } = [];
}