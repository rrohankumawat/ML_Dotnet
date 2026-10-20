using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 0);

var rows = new List<Experience>
{
    new() { Years = 1,  Salary = 30 }, new() { Years = 2,  Salary = 35 },
    new() { Years = 3,  Salary = 41 }, new() { Years = 4,  Salary = 46 },
    new() { Years = 5,  Salary = 52 }, new() { Years = 6,  Salary = 57 },
    new() { Years = 7,  Salary = 63 }, new() { Years = 8,  Salary = 68 },
    new() { Years = 9,  Salary = 74 }, new() { Years = 10, Salary = 79 },
};
IDataView data = mlContext.Data.LoadFromEnumerable(rows);

var pipeline = mlContext.Transforms
    .Concatenate("Features", nameof(Experience.Years))
    .Append(mlContext.Regression.Trainers.Ols(
        labelColumnName: nameof(Experience.Salary),
        featureColumnName: "Features"));

ITransformer model = pipeline.Fit(data);

var engine = mlContext.Model.CreatePredictionEngine<Experience, SalaryPrediction>(model);
var result = engine.Predict(new Experience { Years = 12 });
Console.WriteLine($"12 years of experience -> about {result.Salary:F1}k");

public class Experience
{
    public float Years { get; set; }
    public float Salary { get; set; }
}

public class SalaryPrediction
{
    [ColumnName("Score")]
    public float Salary { get; set; }
}