using Microsoft.ML;
using Microsoft.ML.Data;

const string ModelFile = "models/model.onnx";
const string LabelsFile = "models/synset.txt";
const string ImageFolder = "images";
const string ImageName = "cat.jpg";
const string InputName = "input";   
const string OutputName = "output";  
const int PixelsPerColour = 224 * 224;

float[] mean = { 0.485f, 0.456f, 0.406f };
float[] std = { 0.229f, 0.224f, 0.225f };

var mlContext = new MLContext();

var pipeline = mlContext.Transforms
    .LoadImages("ImageObject", ImageFolder, nameof(ImageInput.ImagePath))
    .Append(mlContext.Transforms.ResizeImages("ImageObject", 224, 224))
    .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject", interleavePixelColors: false, scaleImage: 1f / 255f))
    .Append(mlContext.Transforms.CustomMapping<PixelData, ModelInput>((src, dst) =>
    {
        dst.Input = new float[src.Pixels.Length];
        for (int i = 0; i < src.Pixels.Length; i++)
        {
            int colour = i / PixelsPerColour;  
            dst.Input[i] = (src.Pixels[i] - mean[colour]) / std[colour];
        }
    }, contractName: null))
    .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: ModelFile,
    outputColumnNames: new[] { OutputName },inputColumnNames: new[] { InputName }));

var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<ImageInput>()));

var data = mlContext.Data.LoadFromEnumerable(new[] { new ImageInput { ImagePath = ImageName } });
float[] scores = model.Transform(data).GetColumn<float[]>(OutputName).First();

float best = scores.Max();
double[] exp = scores.Select(s => Math.Exp(s - best)).ToArray();
double total = exp.Sum();

string[] labels = File.ReadAllLines(LabelsFile).Select(line => line[(line.IndexOf(' ') + 1)..]).ToArray();

Console.WriteLine("Top 5 predictions:");
foreach (int position in Enumerable.Range(0, scores.Length).OrderByDescending(p => scores[p]).Take(5))
{
    Console.WriteLine($"  {exp[position] / total,6:P1}  {labels[position]}");
}

public class ImageInput
{
    public string ImagePath { get; set; } = "";
}

public class PixelData
{
    [VectorType(3 * 224 * 224)]
    public float[] Pixels { get; set; } = [];
}

public class ModelInput
{
    [ColumnName("input"), VectorType(3 * 224 * 224)]
    public float[] Input { get; set; } = [];
}