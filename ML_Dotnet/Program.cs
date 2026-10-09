
using Microsoft.ML;
using Microsoft.ML.Data;

var mlcontext = new MLContext(seed: 0);

string filepath = AppContext.BaseDirectory + "reviews.tsv";

var data = mlcontext.Data.LoadFromTextFile<SentimentData>(filepath, new TextLoader.Options
{
    HasHeader = false,
    Separators = new[] {'\t'}
});

var splits = mlcontext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 2);

var pipeline = mlcontext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text)).Append(
                mlcontext.BinaryClassification.Trainers.
                SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features")
    );

Console.WriteLine("Training...");

var model = pipeline.Fit(splits.TrainSet);

var predictionsset = model.Transform(splits.TestSet);

var metrix = mlcontext.BinaryClassification.Evaluate(predictionsset, labelColumnName: "Label");

Console.WriteLine($"Accuracy: {metrix.Accuracy:P1}");
Console.WriteLine($"F1 Score: {metrix.F1Score:P1}");
Console.WriteLine($"AUC: {metrix.AreaUnderRocCurve:P1}");

mlcontext.Model.Save(model, data.Schema, "sentiment.zip");

var loadedmodel = mlcontext.Model.Load("sentiment.zip", out _);

var engine = mlcontext.Model.CreatePredictionEngine<SentimentData, SentimentPredictions>(loadedmodel);

string[] tests = ["This place is amazing, i'll visit again", "Coffee taste is bad and staff behaviour is bad.", "pizza is not good but coffee was good"];

foreach(var  test in tests)
{
    var p = engine.Predict(new SentimentData { Text = test });

    float confidence = p.IsPositive ? p.Probability : 1 - p.Probability;

    Console.WriteLine($"This review is {(p.IsPositive ? "Positive" : "Negative")}. {confidence:P0}, {test}");
}

public class SentimentData
{
    [LoadColumn(0)]
    public string Text { get; set; }
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}

public class SentimentPredictions
{
    [ColumnName("PredictedLabel")]
    public bool IsPositive { get; set; }
    public float Score { get; set; }
    public float Probability { get; set; }
}