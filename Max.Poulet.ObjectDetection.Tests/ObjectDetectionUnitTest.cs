using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Xunit;

namespace Max.Poulet.ObjectDetection.Tests;

public class ObjectDetectionUnitTest
{
    [Fact]
    public async Task ObjectShouldBeDetectedCorrectly()
    {
        var executingPath = GetExecutingPath();
        var imageScenesData = new List<byte[]>();

        foreach (var imagePath in Directory.EnumerateFiles(Path.Combine(executingPath, "Scenes")))
        {
            var imageBytes = await File.ReadAllBytesAsync(imagePath);
            imageScenesData.Add(imageBytes);
        }

        var detectObjectInScenesResults = await new ObjectDetection().DetectObjectInScenesAsync(imageScenesData);

        Assert.Equal(
            "[{\"Label\":\"person\",\"Confidence\":0.6048466,\"Dimensions\":{\"X\":35.485794,\"Y\":49.823624,\"Width\":160.08469,\"Height\":326.36404}},{\"Label\":\"person\",\"Confidence\":0.46399,\"Dimensions\":{\"X\":247.01204,\"Y\":190.85773,\"Width\":115.59238,\"Height\":209.10431}},{\"Label\":\"person\",\"Confidence\":0.4356046,\"Dimensions\":{\"X\":170.56395,\"Y\":56.08972,\"Width\":87.30872,\"Height\":237.28415}},{\"Label\":\"person\",\"Confidence\":0.39464208,\"Dimensions\":{\"X\":168.51767,\"Y\":138.65027,\"Width\":208.99211,\"Height\":268.6819}},{\"Label\":\"bottle\",\"Confidence\":0.37579715,\"Dimensions\":{\"X\":297.51978,\"Y\":4.279972,\"Width\":77.49895,\"Height\":89.05065}}]",
            JsonSerializer.Serialize(detectObjectInScenesResults[0].Box)
        );

        Assert.Equal(
            "[{\"Label\":\"pottedplant\",\"Confidence\":0.31638098,\"Dimensions\":{\"X\":221.53369,\"Y\":64.10928,\"Width\":111.709724,\"Height\":220.88132}}]",
            JsonSerializer.Serialize(detectObjectInScenesResults[1].Box)
        );
        
        var executingPathtest = GetExecutingPath();
    }

    private static string GetExecutingPath()
    {
        var executingAssemblyPath = Assembly.GetExecutingAssembly().Location;
        var executingPath = Path.GetDirectoryName(executingAssemblyPath);
        return executingPath;
    }
}