using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.Security.Cryptography;

var mlContext = new MLContext(seed: 0);

var random = new Random(15);

var ratings = new List<Rating>();

for(int user = 1; user <= 40; user++)
{
    for(int movie = 1; movie <= 20; movie++)
    {
        if (random.NextDouble() < 0.4) continue;

        bool sametaste = user % 2 == movie % 2;

        var movierating = sametaste ? random.Next(4, 6) : random.Next(1, 3);

        ratings.Add(new Rating { Label = movierating, UserId = user , MovieId = movie});

    }
}

var data = mlContext.Data.LoadFromEnumerable<Rating>(ratings);

var splits = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 10);

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "UserIdKey",
    MatrixRowIndexColumnName = "MovieIdKey",
    LabelColumnName = "Label",
    NumberOfIterations = 40,
    ApproximationRank = 8
};

var pipeline = mlContext.Transforms.Conversion.MapValueToKey("UserIdKey", nameof(Rating.UserId)).
    Append(mlContext.Transforms.Conversion.MapValueToKey("MovieIdKey", nameof(Rating.MovieId))).
    Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

var model = pipeline.Fit(splits.TrainSet);

var transforms = model.Transform(splits.TestSet);


var evaluation = mlContext.Regression.Evaluate(transforms, labelColumnName: "Label");


Console.WriteLine(evaluation.RootMeanSquaredError);


public class Rating
{
    public float UserId { get; set; }
    public float MovieId { get; set; }
    public float Label { get; set; }
}