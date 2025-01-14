using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ObjectDetection;

namespace Max.Poulet.ObjectDetection;

public class ObjectDetection
{
    public async Task<IList<ObjectDetectionResult>> DetectObjectInScenesAsync(IList<byte[]> imagesSceneData)
    {
        // Initialisation de Yolo
        var tinyYolo = new Yolo();

        // Liste des tâches parallèles pour traiter chaque image
        var tasks = imagesSceneData.Select(imageData =>
            Task.Run(() =>
            {
                // Appel de la méthode Detect pour chaque image
                var detectionResults = tinyYolo.Detect(imageData);

                // Transformation des résultats en ObjectDetectionResult
                return new ObjectDetectionResult
                {
                    ImageData = detectionResults.ImageData, // Image avec les boîtes dessinées
                    Box = detectionResults.Boxes.Select(box => new BoundingBox
                    {
                        Label = box.Label,
                        Confidence = box.Confidence,
                        Dimensions = new BoundingBoxDimensions
                        {
                            X = box.Dimensions.X,
                            Y = box.Dimensions.Y,
                            Width = box.Dimensions.Width,
                            Height = box.Dimensions.Height
                        }
                    }).ToList()
                };
            })
        );

        // Attendre que toutes les tâches se terminent
        var results = await Task.WhenAll(tasks);

        // Retourner la liste des résultats
        return results.ToList();
    }
}