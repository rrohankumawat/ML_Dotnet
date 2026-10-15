
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;

var mlContext = new MLContext(seed: 0);
var random = new Random(15);

var ratings = new List<Rating>();

for (int user = 1; user <= 40; user++)
{
    for (int movie = 1; movie <= 20; movie++)
    {
        if (random.NextDouble() < 0.4) continue;

        bool sameTaste = user % 2 == movie % 2;

        float movieRating = sameTaste? random.Next(4, 6): random.Next(1, 3);

        ratings.Add(new Rating
        {
            Label = movieRating,
            UserId = user,
            MovieId = movie
        });
    }
}

var data = mlContext.Data.LoadFromEnumerable(ratings);

var splits = mlContext.Data.TrainTestSplit(data,testFraction: 0.2,seed: 10);

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "UserIdKey",
    MatrixRowIndexColumnName = "MovieIdKey",
    LabelColumnName = nameof(Rating.Label),
    NumberOfIterations = 40,
    ApproximationRank = 8
};

var pipeline = mlContext.Transforms.Conversion
    .MapValueToKey("UserIdKey", nameof(Rating.UserId))
    .Append(
        mlContext.Transforms.Conversion.MapValueToKey(
            "MovieIdKey", nameof(Rating.MovieId)))
    .Append(
        mlContext.Recommendation()
            .Trainers.MatrixFactorization(options));

var model = pipeline.Fit(splits.TrainSet);

var testPredictions = model.Transform(splits.TestSet);

var metrics = mlContext.Regression.Evaluate(
    testPredictions,
    labelColumnName: nameof(Rating.Label),
    scoreColumnName: "Score");

Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:F4}");

var engine =mlContext.Model.CreatePredictionEngine<Rating, RatingPrediction>(model);

int userId = 100;

var trainRatings = mlContext.Data.CreateEnumerable<Rating>(splits.TrainSet,reuseRowObject: false).ToList();

bool userKnown = trainRatings.Any(r => r.UserId == userId);

IEnumerable<(int MovieId, float Score)> picks;

if (userKnown)
{
    var seenMovies = trainRatings.Where(r => r.UserId == userId).Select(r => (int)r.MovieId).ToHashSet();

    picks = Enumerable.Range(1, 20)
        .Where(movieId => !seenMovies.Contains(movieId))
        .Select(movieId =>
        {
            var prediction = engine.Predict(new Rating
            {
                UserId = userId,
                MovieId = movieId
            });

            return (
                MovieId: movieId,
                Score: prediction.Score
            );
        })
        .OrderByDescending(x => x.Score)
        .Take(3)
        .ToList();

    Console.WriteLine($"Personalized recommendations for user {userId}:");
}
else
{
    picks = ratings
        .GroupBy(r => (int)r.MovieId)
        .Where(g => g.Count() >= 5)
        .Select(g => (
            MovieId: g.Key,
            Score: g.Average(r => r.Label)
        ))
        .OrderByDescending(x => x.Score)
        .ThenBy(x => x.MovieId)
        .Take(3)
        .ToList();

    Console.WriteLine($"Fallback recommendations for user {userId}:");
}

foreach (var pick in picks)
{
    Console.WriteLine($"Movie ID: {pick.MovieId}, Score: {pick.Score:F2}");
}

public class Rating
{
    public float UserId { get; set; }
    public float MovieId { get; set; }
    public float Label { get; set; }
}

public class RatingPrediction
{
    public float Score { get; set; }
}
