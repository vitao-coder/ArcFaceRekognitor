using ArcFaceRekognitor.Api.FaceRecognition;
using ArcFaceRekognitor.Api.Models;
using ArcFaceRekognitor.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// Additional configuration is required to successfully run gRPC on macOS.
// For instructions on how to configure Kestrel and gRPC clients on macOS, visit https://go.microsoft.com/fwlink/?linkid=2099682

// Add services to the container.
builder.Services.AddSingleton<ModelProvider>();
builder.Services.AddTransient<FaceRecognize>();
builder.Services.AddGrpc();

// Configure Kestrel to listen on a specific HTTP port 
builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenAnyIP(443);
    options.ListenAnyIP(8080, listenOptions =>
    {
        listenOptions.Protocols = Microsoft.AspNetCore.Server.Kestrel.Core.HttpProtocols.Http2;
    });
});

var app = builder.Build();
app.Services.GetService<ModelProvider>();
// Configure the HTTP request pipeline.
app.MapGrpcService<InferenceService>();
app.MapGet("/", () => "Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");


app.Run();
