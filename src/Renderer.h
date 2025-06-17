#ifndef RENDERER_H
#define RENDERER_H

#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "KohonenNetwork.h"

class Renderer {
public:
    static Renderer* instance;

    Renderer(int width, int height);
    ~Renderer();

    bool initialize(int argc, char** argv);
    void setNetwork(const KohonenNetwork* network) { this->network = network; }
    void startMainLoop();

    void updateCamera();

private:
    int windowWidth, windowHeight;
    const KohonenNetwork* network;

    float cameraDistance;
    float cameraAngleX, cameraAngleY;
    bool mousePressed;
    int lastMouseX, lastMouseY;

    void drawCubeWireframe();
    void drawSphere(float x, float y, float z, float r, float g, float b, float radius = 0.08f);
    void drawTexturedSphere(float x, float y, float z, const std::vector<float>& imageData, float radius = 0.08f);
    void drawMNISTImageOnSphere(float x, float y, float z, const std::vector<float>& imageData, float radius = 0.08f);

    bool showImages;

    GLuint createTextureFromMNIST(const std::vector<float>& imageData);

    static void display();
    static void reshape(int width, int height);
    static void idle();
    static void mouseButton(int button, int state, int x, int y);
    static void mouseMotion(int x, int y);
    static void keyboard(unsigned char key, int x, int y);
    static void specialKeys(int key, int x, int y);
};

#endif
