using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 0);

var random = new Random(42);
string[] cities = ["Austin", "Denver", "Seatle"];

var houses = Enumerable.Range(1, 600).Select(_ =>
{
    var city = cities[random.Next(cities.Length)];
    var age = random.Next(1, 30);
    var bedroom = random.Next(1, 6);
    var squarefeet = random.Next(600, 4000);
    var citybonus = city switch
    {
        "Seatle" => 120000,
        "Denver" => 60000,
        _ => 30000
    };

    float noise = random.Next(-20000, 20000);

    float price = 50000 + squarefeet * 150 + bedroom * 10000 - age * 800 + noise + citybonus;

    return new House
    {
        City = city,
        Age = age,
        Bedroom = bedroom,
        Price = price,
        SqareFeet = squarefeet,
    };


}).ToList();

var data = mlContext.Data.LoadFromEnumerable<House>(houses);

var splits = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 2);

var pipleline = mlContext.Transforms.Categorical.OneHotEncoding("CityEncoded", nameof(House.City)).
                Append(mlContext.Transforms.Concatenate("Features", "CityEncoded", nameof(House.Age),
                nameof(House.Bedroom), nameof(House.SqareFeet))).
                Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: nameof(House.Price), featureColumnName: "Features"));

var model = pipleline.Fit(splits.TrainSet);

var predictions = model.Transform(splits.TestSet);

var evaluation = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(House.Price));


Console.WriteLine($"R-Square: {evaluation.RSquared}");
Console.WriteLine($"Mean Absolute Error: {evaluation.MeanAbsoluteError}");
Console.WriteLine($"Root Mean Square Error: {evaluation.RootMeanSquaredError}");

var engine = mlContext.Model.CreatePredictionEngine<House, PricePrediction>(model);

var predict = engine.Predict(new House { Age = 30, Bedroom = 4, City = "Seatle", SqareFeet = 2100 });

Console.WriteLine(predict.Price);

public class House
{
    public string City { get; set; } = "";
    public float Age { get; set; }
    public float Bedroom { get; set; }
    public float Price { get; set; }
    public float SqareFeet { get; set; }
}


public class PricePrediction 
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
