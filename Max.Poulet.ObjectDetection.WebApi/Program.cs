using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.MapGet("/weatherforecast", () =>
{
    var forecast =  Enumerable.Range(1, 5).Select(index =>
        new WeatherForecast
        (
            DateOnly.FromDateTime(DateTime.Now.AddDays(index)),
            Random.Shared.Next(-20, 55),
            summaries[Random.Shared.Next(summaries.Length)]
        ))
        .ToArray();
    return forecast;
})
.WithName("GetWeatherForecast")
.WithOpenApi();

app.MapPost("/ObjectDetection", async ([FromForm] IFormFileCollection files) =>
{
    // Vérification des fichiers envoyés
    if (files.Count < 1)
        return Results.BadRequest("A scene image file is required.");

    // Lecture du fichier image de la scène
    using var sceneSourceStream = files[0].OpenReadStream();
    using var sceneMemoryStream = new MemoryStream();
    await sceneSourceStream.CopyToAsync(sceneMemoryStream);
    var imageSceneData = sceneMemoryStream.ToArray();

    // Utilisation de la librairie pour détecter les objets
    var objectDetection = new Max.Poulet.ObjectDetection.ObjectDetection();
    var detectionResults = await objectDetection.DetectObjectInScenesAsync(new[] { imageSceneData });

    // Vérifier qu'un résultat est disponible
    if (detectionResults == null || detectionResults.Count == 0)
        return Results.BadRequest("No objects were detected.");

    // Récupérer les données de l'image détectée
    var resultImageData = detectionResults[0].ImageData;

    // Retourner l'image avec les zones marquées
    return Results.File(resultImageData, "image/jpg");
}).DisableAntiforgery();

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}
