#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "include/NvFlex.h"
#include "include/NvFlexExt.h"
#include "include/NvFlexDevice.h"

#include "core/maths.h"
#include "core/types.h"
#include "core/platform.h"
#include "core/mesh.h"
#include "core/voxelize.h"
#include "core/sdf.h"
#include "core/pfm.h"
#include "core/tga.h"
#include "core/perlin.h"
#include "core/convex.h"
#include "core/cloth.h"

#include "external/SDL2-2.0.4/include/SDL.h"

#include "shaders.h"
#include "imgui.h"
#include "shadersDemoContext.h"

#include "bindings/utils/utils.h"

void InitRenderHeadless(const RenderInitOptions &options, int width, int height);

SDL_Window *g_window;    // window handle
unsigned int g_windowId; // window id

using namespace std;

int g_screenWidth = 720;
int g_screenHeight = 720;
int g_msaaSamples = 8;

int g_numSubsteps;

// a setting of -1 means Flex will use the device specified in the NVIDIA control panel
int g_device = -1;
bool g_vsync = true;

bool g_extensions = true;
bool g_teamCity = false;
bool g_interop = true;
bool g_d3d12 = false;
bool g_useAsyncCompute = true;
bool g_increaseGfxLoadForAsyncComputeTesting = false;
int g_graphics = 0; // 0=ogl, 1=DX11, 2=DX12

FluidRenderer *g_fluidRenderer;
FluidRenderBuffers *g_fluidRenderBuffers;
DiffuseRenderBuffers *g_diffuseRenderBuffers;

NvFlexSolver *g_solver;
NvFlexSolverDesc g_solverDesc;
NvFlexLibrary *g_flexLib;
NvFlexParams g_params;
NvFlexTimers g_timers;
int g_numDetailTimers;
NvFlexDetailTimer *g_detailTimers;

int g_maxDiffuseParticles;
int g_maxNeighborsPerParticle;
int g_numExtraParticles;
int g_numExtraMultiplier = 1;
int g_maxContactsPerParticle;

// mesh used for deformable object rendering
Mesh *g_mesh;
vector<Mesh *> g_meshList;
vector<int> g_meshSkinIndices;
vector<float> g_meshSkinWeights;
vector<Point3> g_meshRestPositions;
const int g_numSkinWeights = 4;

// mapping of collision mesh to render mesh
std::map<NvFlexConvexMeshId, GpuMesh *> g_convexes;
std::map<NvFlexTriangleMeshId, GpuMesh *> g_meshes;
std::map<NvFlexDistanceFieldId, GpuMesh *> g_fields;

// flag to request collision shapes be updated
bool g_shapesChanged = false;

/* Note that this array of colors is altered by demo code, and is also read from global by graphics API impls */
Colour g_colors[] = {
    Colour(0.000f, 0.500f, 1.000f),
    Colour(0.875f, 0.782f, 0.051f),
    Colour(0.800f, 0.100f, 0.100f),
    Colour(0.673f, 0.111f, 0.000f),
    Colour(0.612f, 0.194f, 0.394f),
    Colour(0.0f, 1.f, 0.0f),
    Colour(0.797f, 0.354f, 0.000f),
    Colour(0.092f, 0.465f, 0.820f)};

//Colour g_colors[] = {
//        Colour(0.0f, 0.0f, 0.0f),
//        Colour(0.05f, 0.05f, 0.05f),
//        Colour(0.1f, 0.1f, 0.1f),
//        Colour(0.15f, 0.15f, 0.15f),
//        Colour(0.2f, 0.2f, 0.2f),
//        Colour(0.25f, 0.25f, 0.25f),
//        Colour(0.3f, 0.3f, 0.3f),
//        Colour(0.35f, 0.35f, 0.35f),
//        Colour(0.4f, 0.4f, 0.4f),
//        Colour(0.45f, 0.45f, 0.45f),
//        Colour(0.5f, 0.5f, 0.5f),
//        Colour(0.55f, 0.55f, 0.55f),
//        Colour(0.6f, 0.6f, 0.6f),
//        Colour(0.65f, 0.65f, 0.65f),
//        Colour(0.7f, 0.7f, 0.7f),
//        Colour(0.75f, 0.75f, 0.75f),
//        Colour(0.8f, 0.8f, 0.8f),
//        Colour(0.85f, 0.85f, 0.85f),
//        Colour(0.9f, 0.9f, 0.9f),
//        Colour(0.95f, 0.95f, 0.95f),
//};

struct SimBuffers
{
    NvFlexVector<Vec4> positions;
    NvFlexVector<Vec4> restPositions;
    NvFlexVector<Vec3> velocities;
    NvFlexVector<int> phases;
    NvFlexVector<float> densities;
    NvFlexVector<Vec4> anisotropy1;
    NvFlexVector<Vec4> anisotropy2;
    NvFlexVector<Vec4> anisotropy3;
    NvFlexVector<Vec4> normals;
    NvFlexVector<Vec4> smoothPositions;
    NvFlexVector<Vec4> diffusePositions;
    NvFlexVector<Vec4> diffuseVelocities;
    NvFlexVector<int> diffuseCount;

    NvFlexVector<int> activeIndices;

    // convexes
    NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
    NvFlexVector<Vec4> shapePositions;
    NvFlexVector<Quat> shapeRotations;
    NvFlexVector<Vec4> shapePrevPositions;
    NvFlexVector<Quat> shapePrevRotations;
    NvFlexVector<int> shapeFlags;

    // rigids
    NvFlexVector<int> rigidOffsets;
    NvFlexVector<int> rigidIndices;
    NvFlexVector<int> rigidMeshSize;
    NvFlexVector<float> rigidCoefficients;
    NvFlexVector<float> rigidPlasticThresholds;
    NvFlexVector<float> rigidPlasticCreeps;
    NvFlexVector<Quat> rigidRotations;
    NvFlexVector<Vec3> rigidTranslations;
    NvFlexVector<Vec3> rigidLocalPositions;
    NvFlexVector<Vec4> rigidLocalNormals;

    // inflatables
    NvFlexVector<int> inflatableTriOffsets;
    NvFlexVector<int> inflatableTriCounts;
    NvFlexVector<float> inflatableVolumes;
    NvFlexVector<float> inflatableCoefficients;
    NvFlexVector<float> inflatablePressures;

    // springs
    NvFlexVector<int> springIndices;
    NvFlexVector<float> springLengths;
    NvFlexVector<float> springStiffness;

    NvFlexVector<int> triangles;
    NvFlexVector<Vec3> triangleNormals;
    NvFlexVector<Vec3> uvs;

    SimBuffers(NvFlexLibrary *l) : positions(l), restPositions(l), velocities(l), phases(l), densities(l),
                                   anisotropy1(l), anisotropy2(l), anisotropy3(l), normals(l), smoothPositions(l),
                                   diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l),
                                   shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
                                   shapePrevRotations(l), shapeFlags(l), rigidOffsets(l), rigidIndices(l), rigidMeshSize(l),
                                   rigidCoefficients(l), rigidPlasticThresholds(l), rigidPlasticCreeps(l), rigidRotations(l),
                                   rigidTranslations(l),
                                   rigidLocalPositions(l), rigidLocalNormals(l), inflatableTriOffsets(l),
                                   inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
                                   inflatablePressures(l), springIndices(l), springLengths(l),
                                   springStiffness(l), triangles(l), triangleNormals(l), uvs(l) {}
};

SimBuffers *g_buffers;

void MapBuffers(SimBuffers *buffers)
{
    buffers->positions.map();
    buffers->restPositions.map();
    buffers->velocities.map();
    buffers->phases.map();
    buffers->densities.map();
    buffers->anisotropy1.map();
    buffers->anisotropy2.map();
    buffers->anisotropy3.map();
    buffers->normals.map();
    buffers->diffusePositions.map();
    buffers->diffuseVelocities.map();
    buffers->diffuseCount.map();
    buffers->smoothPositions.map();
    buffers->activeIndices.map();

    // convexes
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();

    buffers->rigidOffsets.map();
    buffers->rigidIndices.map();
    buffers->rigidMeshSize.map();
    buffers->rigidCoefficients.map();
    buffers->rigidPlasticThresholds.map();
    buffers->rigidPlasticCreeps.map();
    buffers->rigidRotations.map();
    buffers->rigidTranslations.map();
    buffers->rigidLocalPositions.map();
    buffers->rigidLocalNormals.map();

    buffers->springIndices.map();
    buffers->springLengths.map();
    buffers->springStiffness.map();

    // inflatables
    buffers->inflatableTriOffsets.map();
    buffers->inflatableTriCounts.map();
    buffers->inflatableVolumes.map();
    buffers->inflatableCoefficients.map();
    buffers->inflatablePressures.map();

    buffers->triangles.map();
    buffers->triangleNormals.map();
    buffers->uvs.map();
}

void UnmapBuffers(SimBuffers *buffers)
{
    // particles
    buffers->positions.unmap();
    buffers->restPositions.unmap();
    buffers->velocities.unmap();
    buffers->phases.unmap();
    buffers->densities.unmap();
    buffers->anisotropy1.unmap();
    buffers->anisotropy2.unmap();
    buffers->anisotropy3.unmap();
    buffers->normals.unmap();
    buffers->diffusePositions.unmap();
    buffers->diffuseVelocities.unmap();
    buffers->diffuseCount.unmap();
    buffers->smoothPositions.unmap();
    buffers->activeIndices.unmap();

    // convexes
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();

    // rigids
    buffers->rigidOffsets.unmap();
    buffers->rigidIndices.unmap();
    buffers->rigidMeshSize.unmap();
    buffers->rigidCoefficients.unmap();
    buffers->rigidPlasticThresholds.unmap();
    buffers->rigidPlasticCreeps.unmap();
    buffers->rigidRotations.unmap();
    buffers->rigidTranslations.unmap();
    buffers->rigidLocalPositions.unmap();
    buffers->rigidLocalNormals.unmap();

    // springs
    buffers->springIndices.unmap();
    buffers->springLengths.unmap();
    buffers->springStiffness.unmap();

    // inflatables
    buffers->inflatableTriOffsets.unmap();
    buffers->inflatableTriCounts.unmap();
    buffers->inflatableVolumes.unmap();
    buffers->inflatableCoefficients.unmap();
    buffers->inflatablePressures.unmap();

    // triangles
    buffers->triangles.unmap();
    buffers->triangleNormals.unmap();
    buffers->uvs.unmap();
}

SimBuffers *AllocBuffers(NvFlexLibrary *lib)
{
    return new SimBuffers(lib);
}

void DestroyBuffers(SimBuffers *buffers)
{
    // particles
    buffers->positions.destroy();
    buffers->restPositions.destroy();
    buffers->velocities.destroy();
    buffers->phases.destroy();
    buffers->densities.destroy();
    buffers->anisotropy1.destroy();
    buffers->anisotropy2.destroy();
    buffers->anisotropy3.destroy();
    buffers->normals.destroy();
    buffers->diffusePositions.destroy();
    buffers->diffuseVelocities.destroy();
    buffers->diffuseCount.destroy();
    buffers->smoothPositions.destroy();
    buffers->activeIndices.destroy();

    // convexes
    buffers->shapeGeometry.destroy();
    buffers->shapePositions.destroy();
    buffers->shapeRotations.destroy();
    buffers->shapePrevPositions.destroy();
    buffers->shapePrevRotations.destroy();
    buffers->shapeFlags.destroy();

    // rigids
    buffers->rigidOffsets.destroy();
    buffers->rigidIndices.destroy();
    buffers->rigidMeshSize.destroy();
    buffers->rigidCoefficients.destroy();
    buffers->rigidPlasticThresholds.destroy();
    buffers->rigidPlasticCreeps.destroy();
    buffers->rigidRotations.destroy();
    buffers->rigidTranslations.destroy();
    buffers->rigidLocalPositions.destroy();
    buffers->rigidLocalNormals.destroy();

    // springs
    buffers->springIndices.destroy();
    buffers->springLengths.destroy();
    buffers->springStiffness.destroy();

    // inflatables
    buffers->inflatableTriOffsets.destroy();
    buffers->inflatableTriCounts.destroy();
    buffers->inflatableVolumes.destroy();
    buffers->inflatableCoefficients.destroy();
    buffers->inflatablePressures.destroy();

    // triangles
    buffers->triangles.destroy();
    buffers->triangleNormals.destroy();
    buffers->uvs.destroy();

    delete buffers;
}

Vec3 g_camPos(6.0f, 8.0f, 18.0f);
Vec3 g_camAngle(0.0f, -DegToRad(20.0f), 0.0f);
Vec3 g_camVel(0.0f);
Vec3 g_camSmoothVel(0.0f);

float g_camSpeed;
float g_camNear;
float g_camFar;

float g_fov = kPi / 4.0f;

Vec3 g_lightPos;
Vec3 g_lightDir;
Vec3 g_lightTarget;
float g_lightFov;

bool g_pause = false;
bool g_step = false;
bool g_showHelp = true;
bool g_tweakPanel = false;
bool g_fullscreen = false;
bool g_wireframe = false;
bool g_debug = false;

bool g_emit = false;

float g_windTime = 0.0f;
float g_windFrequency = 0.0f;
float g_windStrength = 0.0f;

bool g_wavePool = false;
float g_waveTime = 0.0f;
float g_wavePlane;
float g_waveFrequency = 1.5f;
float g_waveAmplitude = 1.0f;
float g_waveFloorTilt = 0.0f;

Vec3 g_shape_color = Vec3(0.9);
Vec3 g_sceneLower;
Vec3 g_sceneUpper;

float g_blur;
float g_ior;
bool g_drawEllipsoids;
bool g_drawPoints;
bool g_drawMesh;
bool g_drawCloth;
float g_expandCloth; // amount to expand cloth along normal (to account for particle radius)
vector<float> g_particleRadius; // radius for rendering
float g_colorGamma; // make color more natural

bool g_drawOpaque;
int g_drawSprings; // 0: no draw, 1: draw stretch 2: draw tether
bool g_drawBases = false;
bool g_drawContacts = false;
bool g_drawNormals = false;
bool g_drawDiffuse;
bool g_drawShapeGrid = false;
bool g_drawDensity = false;
bool g_drawRopes;
float g_pointScale;
float g_ropeScale;
bool g_drawPlane;
float g_drawPlaneBias; // move planes along their normal for rendering

float g_diffuseScale;
float g_diffuseMotionScale;
bool g_diffuseShadow;
float g_diffuseInscatter;
float g_diffuseOutscatter;


vector<int> g_bodiesParticleOffset;
vector<int> g_bodiesNumParticles;
vector<bool> g_bodiesNeedsSmoothing;
vector<Vec4> g_bodiesColor;
vector<bool> g_bodiesDrawDensity;
vector<bool> g_bodiesDrawDiffuse;
vector<bool> g_bodiesDrawEllipsoids;
vector<bool> g_bodiesDrawPoints;
vector<float> g_bodiesAnisotropyScale;

float g_dt = 1.0f / 240.0f; // the time delta used for simulation
float g_realdt;             // the real world time delta between updates

float g_waitTime;   // the CPU time spent waiting for the GPU
float g_updateTime; // the CPU time spent on Flex
float g_renderTime; // the CPU time spent calling OpenGL to render the scene
// the above times don't include waiting for vsync
float g_simLatency; // the time the GPU spent between the first and last NvFlexUpdateSolver() operation. Because some GPUs context switch, this can include graphics time.

int g_levelScroll;         // offset for level selection scroll area
bool g_resetScene = false; //if the user clicks the reset button or presses the reset key this is set to true;

int g_frame = 0;
int g_numSolidParticles = 0;

int g_mouseParticle = -1;
float g_mouseT = 0.0f;
Vec3 g_mousePos;
float g_mouseMass;
bool g_mousePicked = false;

// mouse
int g_lastx;
int g_lasty;
int g_lastb = -1;

bool g_profile = false;
bool g_outputAllFrameTimes = false;
bool g_asyncComputeBenchmark = false;

ShadowMap *g_shadowMap;

Vec4 g_diffuseColor;
Vec3 g_meshColor;
Vec3 g_clearColor = Vec3(0.0f); // background color
float g_fogDistance;

FILE *g_ffmpeg;

class Scene;

struct Rope
{
    std::vector<int> mIndices;
};

vector<Rope> g_ropes;

inline float sqr(float x) { return x * x; }

#include "helpers.h"
#include "scenes.h"

#include <iostream>
using namespace std;

EmptyScene *g_scene = new EmptyScene("");

Vec4 correct_gamma(Vec4 color)
{
    color.x = pow(color.x, g_colorGamma);
    color.y = pow(color.y, g_colorGamma);
    color.z = pow(color.z, g_colorGamma);
    return color;
}

void Init()
{
    RandInit();
    if (g_solver)
    {
        if (g_buffers)
            DestroyBuffers(g_buffers);

        DestroyFluidRenderBuffers(g_fluidRenderBuffers);
        DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

        for (auto &iter : g_meshes)
        {
            NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        // std::cout << "mesh destroyed" << endl;

        for (auto &iter : g_fields)
        {
            NvFlexDestroyDistanceField(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        for (auto &iter : g_convexes)
        {
            NvFlexDestroyConvexMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        g_fields.clear();
        g_meshes.clear();
        g_convexes.clear();

        NvFlexDestroySolver(g_solver);
        g_solver = nullptr;
    }

    // alloc buffers
    g_buffers = AllocBuffers(g_flexLib);

    // map during initialization
    MapBuffers(g_buffers);

    // std::cout << "buffers mapped" << endl;

    g_buffers->positions.resize(0);
    g_buffers->velocities.resize(0);
    g_buffers->phases.resize(0);

    g_buffers->rigidOffsets.resize(0);
    g_buffers->rigidIndices.resize(0);
    g_buffers->rigidMeshSize.resize(0);
    g_buffers->rigidRotations.resize(0);
    g_buffers->rigidTranslations.resize(0);
    g_buffers->rigidCoefficients.resize(0);
    g_buffers->rigidPlasticThresholds.resize(0);
    g_buffers->rigidPlasticCreeps.resize(0);
    g_buffers->rigidLocalPositions.resize(0);
    g_buffers->rigidLocalNormals.resize(0);

    g_buffers->springIndices.resize(0);
    g_buffers->springLengths.resize(0);
    g_buffers->springStiffness.resize(0);
    g_buffers->triangles.resize(0);
    g_buffers->triangleNormals.resize(0);
    g_buffers->uvs.resize(0);

    g_meshSkinIndices.resize(0);
    g_meshSkinWeights.resize(0);

    g_buffers->shapeGeometry.resize(0);
    g_buffers->shapePositions.resize(0);
    g_buffers->shapeRotations.resize(0);
    g_buffers->shapePrevPositions.resize(0);
    g_buffers->shapePrevRotations.resize(0);
    g_buffers->shapeFlags.resize(0);

    g_ropes.resize(0);

    // remove collision shapes
    delete g_mesh;
    g_mesh = NULL;

    g_frame = 0;
    g_pause = false;

    g_dt = 1.0f / 100.0f;
    g_waveTime = 0.0f;
    g_windTime = 0.0f;
    g_windStrength = 1.0f;

    g_blur = 1.0f;
    g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
    g_drawEllipsoids = false;
    g_drawPoints = true;
    g_drawCloth = true;
    g_expandCloth = 0.0f;

    g_drawOpaque = false;
    g_drawSprings = false;
    g_drawDiffuse = false;
    g_drawMesh = true;
    g_drawRopes = true;
    g_drawDensity = false;
    g_ior = 1.0f;
    g_fogDistance = 0.005f;

    g_camSpeed = 0.075f;
    g_camNear = 0.01f;
    g_camFar = 20.0f;

    g_pointScale = 1.0f;
    g_ropeScale = 1.0f;
    g_drawPlaneBias = -0.01f;
    g_drawPlane = true;
    
    // sim params
    g_params.gravity[0] = 0.0f;
    g_params.gravity[1] = -9.8f;
    g_params.gravity[2] = 0.0f;

    g_params.wind[0] = 0.0f;
    g_params.wind[1] = 0.0f;
    g_params.wind[2] = 0.0f;

    g_params.radius = 0.15f;
    g_params.viscosity = 0.0f;
    g_params.dynamicFriction = 0.0f;
    g_params.staticFriction = 0.0f;
    g_params.particleFriction = 0.0f; // scale friction between particles by default
    g_params.freeSurfaceDrag = 0.0f;
    g_params.drag = 0.0f;
    g_params.lift = 0.0f;
    g_params.numIterations = 3;
    g_params.fluidRestDistance = 0.0f;
    g_params.solidRestDistance = 0.0f;

    g_params.anisotropyScale = 1.0f;
    g_params.anisotropyMin = 0.1f;
    g_params.anisotropyMax = 2.0f;
    g_params.smoothing = 1.0f;

    g_params.dissipation = 0.0f;
    g_params.damping = 0.0f;
    g_params.particleCollisionMargin = 0.0f;
    g_params.shapeCollisionMargin = 0.0f;
    g_params.collisionDistance = 0.0f;
    g_params.sleepThreshold = 0.0f;
    g_params.shockPropagation = 0.0f;
    g_params.restitution = 0.0f;

    g_params.maxSpeed = FLT_MAX;
    g_params.maxAcceleration = 100.0f; // approximately 10x gravity

    g_params.relaxationMode = eNvFlexRelaxationLocal;
    g_params.relaxationFactor = 1.0f;
    g_params.solidPressure = 1.0f;
    g_params.adhesion = 0.0f;
    g_params.cohesion = 0.025f;
    g_params.surfaceTension = 0.0f;
    g_params.vorticityConfinement = 0.0f;
    g_params.buoyancy = 1.0f;
    g_params.diffuseThreshold = 100.0f;
    g_params.diffuseBuoyancy = 1.0f;
    g_params.diffuseDrag = 0.8f;
    g_params.diffuseBallistic = 16;
    g_params.diffuseLifetime = 2.0f;

    g_numSubsteps = 20;

    // planes created after particles
    g_params.numPlanes = 1;

    g_diffuseScale = 0.5f;
    g_diffuseColor = 1.0f;
    g_diffuseMotionScale = 1.0f;
    g_diffuseShadow = false;
    g_diffuseInscatter = 0.8f;
    g_diffuseOutscatter = 0.53f;

    // reset phase 0 particle color to blue
    //    g_colors[0] = Colour(0.0f, 0.5f, 1.0f);

    g_numSolidParticles = 0;

    g_waveFrequency = 1.5f;
    g_waveAmplitude = 1.5f;
    g_waveFloorTilt = 0.0f;
    g_emit = false;

    g_mouseParticle = -1;

    g_maxDiffuseParticles = 0; // number of diffuse particles
    g_maxNeighborsPerParticle = 96;
    g_numExtraParticles = 0; // number of particles allocated but not made active
    g_maxContactsPerParticle = 6;

    g_sceneLower = FLT_MAX;
    g_sceneUpper = -FLT_MAX;

    // initialize solver desc
    NvFlexSetSolverDescDefaults(&g_solverDesc);

    // create scene
    StartGpuWork();
    //    cout<<thread_idx<<endl;
    g_scene->Initialize();
    EndGpuWork();

    uint32_t numParticles = g_buffers->positions.size();
    uint32_t maxParticles = numParticles + g_numExtraParticles * g_numExtraMultiplier;

    if (g_params.solidRestDistance == 0.0f)
        g_params.solidRestDistance = g_params.radius;

    // if fluid present then we assume solid particles have the same radius
    if (g_params.fluidRestDistance > 0.0f)
        g_params.solidRestDistance = g_params.fluidRestDistance;

    // set collision distance automatically based on rest distance if not already set
    if (g_params.collisionDistance == 0.0f)
        g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance) * 0.5f;

    // default particle friction to 10% of shape friction
    if (g_params.particleFriction == 0.0f)
        g_params.particleFriction = g_params.dynamicFriction * 0.1f;

    // add a margin for detecting contacts between particles and shapes
    if (g_params.shapeCollisionMargin == 0.0f)
        g_params.shapeCollisionMargin = g_params.collisionDistance * 0.5f;

    // calculate particle bounds
    Vec3 particleLower, particleUpper;
    GetParticleBounds(particleLower, particleUpper);

    // accommodate shapes
    Vec3 shapeLower, shapeUpper;
    GetShapeBounds(shapeLower, shapeUpper);

    // update collision planes to match flexs
    Vec3 up = Normalize(Vec3(-g_waveFloorTilt, 1.0f, 0.0f));

    (Vec4 &)g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
    (Vec4 &)g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
    (Vec4 &)g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
    (Vec4 &)g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
    (Vec4 &)g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
    (Vec4 &)g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

    g_wavePlane = g_params.planes[2][3];

    g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
    g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
    g_buffers->diffuseCount.resize(1, 0);

    // for fluid rendering these are the Laplacian smoothed positions
    g_buffers->smoothPositions.resize(maxParticles);

    g_buffers->normals.resize(0);
    g_buffers->normals.resize(maxParticles);

    // initialize normals (just for rendering before simulation starts)
    int numTris = g_buffers->triangles.size() / 3;
    for (int i = 0; i < numTris; ++i)
    {
        Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
        Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
        Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

        Vec3 n = Cross(v1 - v0, v2 - v0);

        g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
    }

    for (int i = 0; i < int(maxParticles); ++i)
        g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);

    // std::cout << "normals initialized" << endl;

    // save mesh positions for skinning
    if (g_mesh)
    {
        g_meshRestPositions = g_mesh->m_positions;
    }
    else
    {
        g_meshRestPositions.resize(0);
    }

    g_solverDesc.maxParticles = maxParticles;
    g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
    g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
    g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

    // main create method for the Flex solver
    g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

    // give scene a chance to do some post solver initialization
    g_scene->PostInitialize();

    // create active indices (just a contiguous block for the demo)
    g_buffers->activeIndices.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
        g_buffers->activeIndices[i] = i;

    // resize particle buffers to fit
    g_buffers->positions.resize(maxParticles);
    g_buffers->velocities.resize(maxParticles);
    g_buffers->phases.resize(maxParticles);
    g_buffers->uvs.resize(maxParticles);

    g_buffers->densities.resize(maxParticles);
    g_buffers->anisotropy1.resize(maxParticles);
    g_buffers->anisotropy2.resize(maxParticles);
    g_buffers->anisotropy3.resize(maxParticles);

    // save rest positions
    // g_buffers->restPositions.resize(g_buffers->positions.size());
    g_buffers->restPositions.resize(maxParticles);
    for (int i = 0; i < g_buffers->positions.size(); ++i)
        g_buffers->restPositions[i] = g_buffers->positions[i];

    // builds rigids constraints
    if (g_buffers->rigidOffsets.size())
    {
        assert(g_buffers->rigidOffsets.size() > 1);

        const int numRigids = g_buffers->rigidOffsets.size() - 1;

        /*
        printf("rigidOffsets\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidOffsets.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidOffsets[i]);
        }

        printf("rigidIndices\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidIndices.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidIndices[i]);
        }
         */

        // If the centers of mass for the rigids are not yet computed, this is done here
        // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
        if (g_buffers->rigidTranslations.size() == 0)
        {
            g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
            CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(),
                                        &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0],
                                        &g_buffers->rigidIndices[0], numRigids);
        }

        // calculate local rest space positions
        g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
        CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0],
                                     &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids,
                                     &g_buffers->rigidLocalPositions[0]);

        // set rigidRotations to correct length, probably NULL up until here
        g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
    }

    // unmap so we can start transferring data to GPU
    UnmapBuffers(g_buffers);

    //-----------------------------
    // Send data to Flex

    NvFlexCopyDesc copyDesc;
    copyDesc.dstOffset = 0;
    copyDesc.srcOffset = 0;
    copyDesc.elementCount = numParticles;

    NvFlexSetParams(g_solver, &g_params);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
    NvFlexSetNormals(g_solver, g_buffers->normals.buffer, &copyDesc);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
    NvFlexSetActiveCount(g_solver, numParticles);

    // springs
    if (g_buffers->springIndices.size())
    {
        assert((g_buffers->springIndices.size() & 1) == 0);
        assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

        NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer,
                         g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
    }

    // rigids
    if (g_buffers->rigidOffsets.size())
    {
        NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer,
                        g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer,
                        g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer,
                        g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer,
                        g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1,
                        g_buffers->rigidIndices.size());
    }

    // std::cout << "rigids setup done" << endl;

    // inflatables
    if (g_buffers->inflatableTriOffsets.size())
    {
        NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer,
                             g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer,
                             g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
    }

    // dynamic triangles
    if (g_buffers->triangles.size())
    {
        NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer,
                                  g_buffers->triangles.size() / 3);
    }

    // collision shapes
    if (g_buffers->shapeFlags.size())
    {
        NvFlexSetShapes(
            g_solver,
            g_buffers->shapeGeometry.buffer,
            g_buffers->shapePositions.buffer,
            g_buffers->shapeRotations.buffer,
            g_buffers->shapePrevPositions.buffer,
            g_buffers->shapePrevRotations.buffer,
            g_buffers->shapeFlags.buffer,
            int(g_buffers->shapeFlags.size()));
    }

    // create render buffers
    g_fluidRenderBuffers = CreateFluidRenderBuffers(maxParticles, g_interop);
    g_diffuseRenderBuffers = CreateDiffuseRenderBuffers(g_maxDiffuseParticles, g_interop);

}

/*
void Reset() {
    Init(g_scene, false);
}
*/

void Shutdown()
{
    // free buffers
    if (g_buffers)
        DestroyBuffers(g_buffers);

    for (auto &iter : g_meshes)
    {
        NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto &iter : g_fields)
    {
        NvFlexDestroyDistanceField(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto &iter : g_convexes)
    {
        NvFlexDestroyConvexMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    g_fields.clear();
    g_meshes.clear();
    if (g_solver)
        NvFlexDestroySolver(g_solver);
    if (g_flexLib)
        NvFlexShutdown(g_flexLib);
}

void UpdateCamera()
{
    Vec3 forward(-sinf(g_camAngle.x) * cosf(g_camAngle.y), sinf(g_camAngle.y),
                 -cosf(g_camAngle.x) * cosf(g_camAngle.y));
    Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_camSmoothVel = Lerp(g_camSmoothVel, g_camVel, 0.1f);
    g_camPos += (forward * g_camSmoothVel.z + right * g_camSmoothVel.x + Cross(right, forward) * g_camSmoothVel.y);
    //    cout<<"g_camPos"<<g_camPos[0] << " " << g_camPos[1] << " " << g_camPos[2]<<endl;
    //    cout<<"g_camAngle"<<g_camAngle[0] << " "<< g_camAngle[1]<<" "<<g_camAngle[2]<<endl;
}

void UpdateMouse()
{
    // mouse button is up release particle
    if (g_lastb == -1)
    {
        if (g_mouseParticle != -1)
        {
            // restore particle mass
            g_buffers->positions[g_mouseParticle].w = g_mouseMass;

            // deselect
            g_mouseParticle = -1;
        }
    }

    // mouse went down, pick new particle
    if (g_mousePicked)
    {
        assert(g_mouseParticle == -1);

        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        const int numActive = NvFlexGetActiveCount(g_solver);

        g_mouseParticle = PickParticle(origin, dir, &g_buffers->positions[0], &g_buffers->phases[0], numActive,
                                       g_params.radius * 0.8f, g_mouseT);

        if (g_mouseParticle != -1)
        {
            printf("picked: %d, mass: %f v: %f %f %f\n", g_mouseParticle, g_buffers->positions[g_mouseParticle].w,
                   g_buffers->velocities[g_mouseParticle].x, g_buffers->velocities[g_mouseParticle].y,
                   g_buffers->velocities[g_mouseParticle].z);

            g_mousePos = origin + dir * g_mouseT;
            g_mouseMass = g_buffers->positions[g_mouseParticle].w;
            g_buffers->positions[g_mouseParticle].w = 0.0f; // increase picked particle's mass to force it towards the point
        }

        g_mousePicked = false;
    }

    // update picked particle position
    if (g_mouseParticle != -1)
    {
        Vec3 p = Lerp(Vec3(g_buffers->positions[g_mouseParticle]), g_mousePos, 0.8f);
        Vec3 delta = p - Vec3(g_buffers->positions[g_mouseParticle]);

        g_buffers->positions[g_mouseParticle].x = p.x;
        g_buffers->positions[g_mouseParticle].y = p.y;
        g_buffers->positions[g_mouseParticle].z = p.z;

        g_buffers->velocities[g_mouseParticle].x = delta.x / g_dt;
        g_buffers->velocities[g_mouseParticle].y = delta.y / g_dt;
        g_buffers->velocities[g_mouseParticle].z = delta.z / g_dt;
    }
}

void UpdateWind()
{
    g_windTime += g_dt;

    const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
    const float kNoise = Perlin1D(g_windTime * g_windFrequency, 10, 0.25f);
    Vec3 wind = g_windStrength * kWindDir * Vec3(kNoise, fabsf(kNoise), 0.0f);

    g_params.wind[0] = wind.x;
    g_params.wind[1] = wind.y;
    g_params.wind[2] = wind.z;

    if (g_wavePool)
    {
        g_waveTime += g_dt;
        g_params.planes[2][3] =
            g_wavePlane + (sinf(float(g_waveTime) * g_waveFrequency - kPi * 0.5f) * 0.5f + 0.5f) * g_waveAmplitude;
    }
}

void RenderScene(bool renderUV = false)
{
    const int numParticles = NvFlexGetActiveCount(g_solver);
    const int numDiffuse = g_buffers->diffuseCount[0];

    //---------------------------------------------------
    // use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
    // Flex will take care of any CUDA interop mapping/unmapping during the get() operations

    if (numParticles)
    {
        if (g_drawEllipsoids)
        {
            // if fluid surface rendering then update with smooth positions and anisotropy
            UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                     &g_buffers->smoothPositions[0],
                                     (g_drawDensity) ? &g_buffers->densities[0] : (float *)&g_buffers->phases[0],
                                     &g_buffers->anisotropy1[0],
                                     &g_buffers->anisotropy2[0],
                                     &g_buffers->anisotropy3[0],
                                     g_buffers->positions.size(),
                                     &g_buffers->activeIndices[0],
                                     numParticles);
        }
        else
        {
            // otherwise just send regular positions and no anisotropy
            UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                     &g_buffers->positions[0],
                                     (float *)&g_buffers->phases[0],
                                     nullptr, nullptr, nullptr,
                                     g_buffers->positions.size(),
                                     &g_buffers->activeIndices[0],
                                     numParticles);
        }
    }

    // GPU Render time doesn't include CPU->GPU copy time
    GraphicsTimerBegin();

    if (numDiffuse)
    {
            // copy diffuse particle data from host to GPU render device
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers,
                                       &g_buffers->diffusePositions[0],
                                       &g_buffers->diffuseVelocities[0],
                                       numDiffuse);
    }

    //---------------------------------------
    // setup view and state

    float aspect = float(g_screenWidth) / g_screenHeight;

    Matrix44 proj = ProjectionMatrix(RadToDeg(g_fov), aspect, g_camNear, g_camFar);
    Matrix44 view = RotationMatrix(-g_camAngle.x, Vec3(0.0f, 1.0f, 0.0f)) *
                    RotationMatrix(-g_camAngle.y, Vec3(cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x))) *
                    TranslationMatrix(-Point3(g_camPos));

    //------------------------------------
    // lighting pass
                    
    Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(g_lightFov), 1.0f, 1.0f, 1000.0f);
    Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
    Matrix44 lightTransform = lightPerspective * lightView;


    if (renderUV)
    {
        BindSolidShader(
            g_lightPos,
            g_lightTarget,
            lightTransform,
            g_shadowMap,
            0.0f,
            Vec4(g_clearColor, g_fogDistance));
        SetView(view, proj);
        SetCullMode(true);
        if (g_drawCloth && g_buffers->triangles.size())
        {
            DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0],
                      g_buffers->uvs.size() ? &g_buffers->uvs[0] : nullptr,
                      &g_buffers->triangles[0],
                      g_buffers->triangles.size() / 3,
                      g_buffers->positions.size(), 3,
                      g_expandCloth, true);
        }
        UnbindSolidShader();
    }
    else
    {
        //-------------------------------------
        // shadowing pass

        if (g_meshSkinIndices.size())
            SkinMesh();

        // create shadow maps
        ShadowBegin(g_shadowMap);

        SetView(lightView, lightPerspective);
        SetCullMode(false);

        // give scene a chance to do custom drawing
        g_scene->Draw(1);

        if (g_drawMesh && g_meshList.size() > 0)
        {
            for (Mesh *m : g_meshList)
            {
                DrawMesh(m);
            }
        }

        int shadowParticles = numParticles;
        int shadowParticlesOffset = 0;

        if (!g_drawPoints)
        {
            shadowParticles = 0;

            if (g_drawEllipsoids)
            {
                shadowParticles = numParticles - g_numSolidParticles;
                shadowParticlesOffset = g_numSolidParticles;
            }
        }
        else
        {
            int offset = g_drawMesh ? g_numSolidParticles : 0;

            shadowParticles = numParticles - offset;
            shadowParticlesOffset = offset;
        }

        if (g_buffers->activeIndices.size())
        {
            for (size_t i = 0; i < (size_t)g_bodiesNumParticles.size(); i++)
            {
                DrawPoints(g_fluidRenderBuffers, g_bodiesNumParticles[i], g_bodiesParticleOffset[i], g_particleRadius[i], float(g_screenWidth), aspect, g_fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, correct_gamma(g_bodiesColor[i]), g_drawDensity);
            }

        }


        ShadowEnd();

        //----------------
        // lighting pass

        BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.0f, Vec4(g_clearColor, g_fogDistance));

        SetView(view, proj);
        SetCullMode(true);

        if (g_drawPlane)
            DrawPlanes((Vec4 *)g_params.planes, g_params.numPlanes, g_drawPlaneBias);

        if (g_drawMesh && g_meshList.size() > 0)
        {
            for (Mesh *m : g_meshList)
            {
                DrawMesh(m);
            }
        }

        // give scene a chance to do custom drawing
        g_scene->Draw(0);
            
        UnbindSolidShader();

        for (size_t i = 0; i < (size_t)g_bodiesNumParticles.size(); i++)
        {
            if (g_bodiesDrawEllipsoids[i])
            {
                // first pass of diffuse particles (behind fluid surface)
                if (g_bodiesDrawDiffuse[i])
                {
                    RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, g_particleRadius[i] * g_diffuseScale,
                                float(g_screenWidth), aspect, g_fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform,
                                g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow,
                                false);
                }

                // draw points first, then ellipsoids, since latter could be transparent
                RenderEllipsoids(g_fluidRenderer, g_fluidRenderBuffers, g_bodiesNumParticles[i], g_bodiesParticleOffset[i],
                                g_particleRadius[i], float(g_screenWidth), aspect, g_fov, g_lightPos, g_lightTarget, lightTransform,
                                g_shadowMap, g_bodiesColor[i], g_blur, g_ior, g_drawOpaque);

                // second pass of diffuse particles for particles in front of fluid surface
                if (g_bodiesDrawDiffuse[i])
                {
                    RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, g_particleRadius[i] * g_diffuseScale,
                                    float(g_screenWidth), aspect, g_fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform,
                                    g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow,
                                    true);
                }
            }
            else if (g_bodiesDrawPoints[i])
            {
                DrawPoints(g_fluidRenderBuffers, g_bodiesNumParticles[i], g_bodiesParticleOffset[i], g_particleRadius[i], float(g_screenWidth),
                           aspect, g_fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, correct_gamma(g_bodiesColor[i]), g_drawDensity);
            }
        }
    }

    GraphicsTimerEnd();
}


bool g_Error = false;
void ErrorCallback(NvFlexErrorSeverity severity, const char *msg, const char *file, int line)
{
    printf("Flex: %s - %s:%d\n", msg, file, line);
    g_Error = (severity == eNvFlexLogError);
    //assert(0); asserts are bad for TeamCity
}

void SDLInit(const char *title)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) <
        0) // Initialize SDL's Video subsystem and game controllers
        printf("Unable to initialize SDL");

    unsigned int flags = SDL_WINDOW_RESIZABLE;

    if (g_graphics == 0)
    {
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL;
    }

    g_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                g_screenWidth, g_screenHeight, flags);

    g_windowId = SDL_GetWindowID(g_window);
}
