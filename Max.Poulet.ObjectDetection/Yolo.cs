using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ObjectDetection;

public class ImageNetData
{
    [LoadColumn(0)] public string ImagePath;

    [LoadColumn(1)] public string Label;

    public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
    {
        return Directory
            .GetFiles(imageFolder)
            .Where(filePath => Path.GetExtension(filePath) != ".md")
            .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
    }
}

public class YoloOutput
{
    public byte[] ImageData { get; set; }

    public IList<BoundingBox> Boxes { get; set; }
}

public class Yolo
{
    private static string GetExecutingPath()
    {
        var executingAssemblyPath = Assembly.GetExecutingAssembly().Location;
        var executingPath = Path.GetDirectoryName(executingAssemblyPath);
        return executingPath;
    }

    public YoloOutput Detect(byte[] image)
    {
        string modelFilePath = Path.Combine(GetExecutingPath(), "TinyYolo2_model.onnx");
        // Initialize MLContext
        var mlContext = new MLContext();

        // Load Data
        var streams = new List<byte[]> { image };
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempPath);
        var index = 0;
        foreach (var stream in streams)
        {
            var filename = $"image_{index++}.jpg";
            var tempFile = Path.Combine(tempPath, filename);
            using var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write);
            fileStream.Write(stream, 0, stream.Length);
        }

        var images = ImageNetData.ReadFromFile(tempPath);
        var imageDataView = mlContext.Data.LoadFromEnumerable(images);

        // Create instance of model scorer
        var modelScorer = new OnnxModelScorer(tempPath, modelFilePath, mlContext);

        // Use model to score data
        var probabilities = modelScorer.Score(imageDataView);

        // Post-process model output
        var parser = new YoloOutputParser();

        var boundingBoxes =
            probabilities
                .Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

        // Draw bounding boxes for detected objects in each of the images
        for (var i = 0; i < images.Count(); i++)
        {
            var imageFileName = images.ElementAt(i).Label;
            var detectedObjects = boundingBoxes.ElementAt(i);

            var stream = DrawBoundingBox(tempPath, imageFileName, detectedObjects);

            LogDetectedObjects(imageFileName, detectedObjects);
            
            IList<BoundingBox> boxes = new List<BoundingBox>();
            foreach (var detectedObject in detectedObjects)
            {
                boxes.Add(new BoundingBox
                {
                    Confidence = detectedObject.Confidence,
                    Label = detectedObject.Label,
                    Dimensions = new BoundingBoxDimensions
                    {
                        Height = detectedObject.Dimensions.Height,
                        Width = detectedObject.Dimensions.Width,
                        X = detectedObject.Dimensions.X,
                        Y = detectedObject.Dimensions.Y
                    }
                });
            }
            return new YoloOutput { ImageData = stream, Boxes = boxes };
        }


        return new YoloOutput();
    }

    public byte[] DrawBoundingBox(string inputImageLocation, string imageName,
    IList<YoloBoundingBox> filteredBoundingBoxes)
{
    var imagePath = Path.Combine(inputImageLocation, imageName);
    var image = Image.FromFile(imagePath);

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    foreach (var box in filteredBoundingBoxes)
    {
        // Get Bounding Box Dimensions
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

        // Resize To Image
        x = (uint)(originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth);
        y = (uint)(originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight);
        width = (uint)(originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth);
        height = (uint)(originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight);

        // Bounding Box Text
        var text = $"{box.Label} ({(box.Confidence * 100):0}%)";

        using var thumbnailGraphic = Graphics.FromImage(image);
        thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
        thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
        thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

        // Define Text Options
        var drawFont = new Font("Arial", 14, FontStyle.Bold);
        var size = thumbnailGraphic.MeasureString(text, drawFont);
        var fontBrush = new SolidBrush(Color.Black);
        
        // créer un brush transparent
        var transparentBrush = new SolidBrush(Color.FromArgb(200, box.BoxColor.R, box.BoxColor.G, box.BoxColor.B));
        var atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

        // Define BoundingBox options
        var pen = new Pen(transparentBrush, 4f);
        var colorBrush = new SolidBrush(Color.FromArgb(200, box.BoxColor.R, box.BoxColor.G, box.BoxColor.B));

        // Draw text on image 
        thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width,
            (int)size.Height);
        thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

        // Draw bounding box on image
        thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
    }

    using var ms = new MemoryStream();
    // save this image to a memory stream
    image.Save(ms, ImageFormat.Jpeg);
    
    return ms.ToArray();
}

    private void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
    {
        Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

        foreach (var box in boundingBoxes)
        {
            Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            Console.WriteLine(
                $"Box Location: X={box.Dimensions.X}, Y={box.Dimensions.Y}, Width={box.Dimensions.Width}, Height={box.Dimensions.Height}");
        }

        Console.WriteLine("");
    }
}

    class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }

        public struct TinyYoloModelSettings
        {
            // for checking Tiny yolo2 Model input and  output  parameter names,
            //you can use tools like Netron, 
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string ModelInput = "image";

            // output tensor name
            public const string ModelOutput = "grid";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            IDataView scoredData = model.Transform(testData);

            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);

            return PredictDataUsingModel(data, model);
        }
    }
    
    
    public class DimensionsBase
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }
    
    
    public class BoundingBoxDimensions : DimensionsBase { }

    public class YoloBoundingBox
    {
        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }
        public Color BoxColor { get; set; }
    }
    
    public class BoundingBox
    {
        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }
    }
    
      class YoloOutputParser
    {
        class CellDimensions : DimensionsBase { }

        public const int ROW_COUNT = 13;
        public const int COL_COUNT = 13;
        public const int CHANNEL_COUNT = 125;
        public const int BOXES_PER_CELL = 5;
        public const int BOX_INFO_FEATURE_COUNT = 5;
        public const int CLASS_COUNT = 20;
        public const float CELL_WIDTH = 32;
        public const float CELL_HEIGHT = 32;

        private int channelStride = ROW_COUNT * COL_COUNT;

        private float[] anchors = new float[]
        {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };

        private string[] labels = new string[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };

        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * this.channelStride) + (y * COL_COUNT) + x;
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
                Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
            };
        }

        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            var boxes = new List<YoloBoundingBox>();

            for (int row = 0; row < ROW_COUNT; row++)
            {
                for (int column = 0; column < COL_COUNT; column++)
                {
                    for (int box = 0; box < BOXES_PER_CELL; box++)
                    {
                        var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);

                        float confidence = GetConfidence(yoloModelOutputs, row, column, channel);

                        CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        if (confidence < threshold)
                            continue;

                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);

                        var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;

                        if (topScore < threshold)
                            continue;

                        boxes.Add(new YoloBoundingBox()
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                                Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = labels[topResultIndex],
                            BoxColor = classColors[topResultIndex]
                        });
                    }
                }
            }
            return boxes;
        }
        
        public RectangleF GetRect(YoloBoundingBox yoloBoundingBox)
        {
            var dimensions = yoloBoundingBox.Dimensions;
           return new RectangleF(dimensions.X, dimensions.Y, dimensions.Width, dimensions.Height); 
        }

        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                                .OrderByDescending(b => b.Box.Confidence)
                                .ToList();

            var results = new List<YoloBoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(GetRect(boxA), GetRect(boxB)) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

    }