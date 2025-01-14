using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Max.Poulet.ObjectDetection;

namespace Max.Poulet.ObjectDetection.Console
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Vérifier que le chemin du répertoire est fourni
            if (args.Length < 1)
            {
                System.Console.WriteLine("Usage: dotnet run <scenesDirectoryPath>");
                return;
            }

            var scenesDirectoryPath = args[0];

            // Vérifier que le répertoire existe
            if (!Directory.Exists(scenesDirectoryPath))
            {
                System.Console.WriteLine($"Error: Directory '{scenesDirectoryPath}' not found.");
                return;
            }

            // Charger les images depuis le répertoire
            var sceneImagePaths = Directory.GetFiles(scenesDirectoryPath);
            var sceneImageData = new List<byte[]>();

            foreach (var sceneImagePath in sceneImagePaths)
            {
                var imageData = await File.ReadAllBytesAsync(sceneImagePath);
                sceneImageData.Add(imageData);
            }

            // Créer une instance de la détection d'objets
            var objectDetection = new ObjectDetection();
            
            // Appeler la méthode de détection
            var detectObjectInScenesResults = await objectDetection.DetectObjectInScenesAsync(sceneImageData);

            // Afficher les résultats pour chaque image
            foreach (var objectDetectionResult in detectObjectInScenesResults)
            {
                System.Console.WriteLine($"Box: {JsonSerializer.Serialize(objectDetectionResult.Box)}");
            }
        }
    }
}

