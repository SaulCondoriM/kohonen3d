#include "Renderer.h"
#include <iostream>
#include <cmath>

Renderer* Renderer::instance = nullptr;

Renderer::Renderer(int width, int height)
: windowWidth(width), windowHeight(height), network(nullptr),
cameraDistance(8.0f), cameraAngleX(20.0f), cameraAngleY(45.0f),
mousePressed(false), lastMouseX(0), lastMouseY(0), showImages(true) {
    instance = this;
}

Renderer::~Renderer() {
    instance = nullptr;
}

bool Renderer::initialize(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Kohonen 3D Network - MNIST Visualization");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_TEXTURE_2D);

    GLfloat lightPos[] = {2.0f, 2.0f, 2.0f, 1.0f};
    GLfloat lightAmbient[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat lightDiffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    GLfloat lightSpecular[] = {1.0f, 1.0f, 1.0f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);

    std::cout << "GLUT Renderer initialized successfully" << std::endl;
    return true;
}

void Renderer::startMainLoop() {
    std::cout << "Starting GLUT main loop..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - Mouse drag: Rotate camera" << std::endl;
    std::cout << "  - Arrow keys: Move camera" << std::endl;
    std::cout << "  - +/-: Zoom in/out" << std::endl;
    std::cout << "  - SPACE: Toggle between images and colors" << std::endl;
    std::cout << "  - ESC: Exit" << std::endl;

    glutMainLoop();
}

void Renderer::drawCubeWireframe() {
    glDisable(GL_LIGHTING);
    glColor3f(0.7f, 0.7f, 0.7f);
    glLineWidth(2.0f);
    glutWireCube(2.0f);
    glEnable(GL_LIGHTING);
}

void Renderer::drawSphere(float x, float y, float z, float r, float g, float b, float radius) {
    glPushMatrix();
    glTranslatef(x, y, z);
    glColor3f(r, g, b);
    glutSolidSphere(radius, 16, 16);
    glPopMatrix();
}

void Renderer::drawMNISTImageOnSphere(float x, float y, float z, const std::vector<float>& imageData, float radius) {
    glPushMatrix();
    glTranslatef(x, y, z);

    glDisable(GL_LIGHTING);

    glColor3f(0.2f, 0.2f, 0.2f);
    glutSolidSphere(radius * 0.8f, 8, 8);

    glBegin(GL_QUADS);
    float baseSize = radius * 3.0f; 

    for (int imgY = 0; imgY < 28; imgY += 2) {
        for (int imgX = 0; imgX < 28; imgX += 2) {
            float pixelValue = imageData[imgY * 28 + imgX];
            if (pixelValue > 0.2f) {
                float pixelSize = baseSize / 7.0f; 
                float posX = (imgX/2 - 7) * pixelSize / 7.0f;
                float posY = (7 - imgY/2) * pixelSize / 7.0f;

                if (pixelValue > 0.7f) {
                    glColor3f(1.0f, 1.0f, 0.0f); 
                } else if (pixelValue > 0.4f) {
                    glColor3f(1.0f, 0.5f, 0.0f); 
                } else {
                    glColor3f(0.8f, 0.8f, 0.8f);
                }

                float z_offset = radius * 1.1f;
                glVertex3f(posX, posY, z_offset);
                glVertex3f(posX + pixelSize/7.0f, posY, z_offset);
                glVertex3f(posX + pixelSize/7.0f, posY - pixelSize/7.0f, z_offset);
                glVertex3f(posX, posY - pixelSize/7.0f, z_offset);
            }
        }
    }
    glEnd();

    glEnable(GL_LIGHTING);
    glPopMatrix();
}

void Renderer::display() {
    if (!instance || !instance->network) return;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float camX = instance->cameraDistance * sin(instance->cameraAngleY * M_PI / 180.0f) * cos(instance->cameraAngleX * M_PI / 180.0f);
    float camY = instance->cameraDistance * sin(instance->cameraAngleX * M_PI / 180.0f);
    float camZ = instance->cameraDistance * cos(instance->cameraAngleY * M_PI / 180.0f) * cos(instance->cameraAngleX * M_PI / 180.0f);

    gluLookAt(camX, camY, camZ, 0, 0, 0, 0, 1, 0);

    GLfloat lightPos[] = {2.0f, 2.0f, 2.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    instance->drawCubeWireframe();

    const auto& neurons = instance->network->getNeurons();
    for (const auto& neuron : neurons) {
        if (instance->showImages) {
            // Show MNIST prototype images
            instance->drawMNISTImageOnSphere(neuron.x, neuron.y, neuron.z,
                                             neuron.prototypeImage);
        } else {
            // Show colored spheres
            instance->drawSphere(neuron.x, neuron.y, neuron.z,
                                 neuron.color[0], neuron.color[1], neuron.color[2]);
        }
    }

    glutSwapBuffers();
}

void Renderer::reshape(int width, int height) {
    if (!instance) return;

    instance->windowWidth = width;
    instance->windowHeight = height;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float aspect = (float)width / (float)height;
    gluPerspective(45.0f, aspect, 0.1f, 100.0f);

    glMatrixMode(GL_MODELVIEW);
}

void Renderer::idle() {
    glutPostRedisplay();
}

void Renderer::mouseButton(int button, int state, int x, int y) {
    if (!instance) return;

    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            instance->mousePressed = true;
            instance->lastMouseX = x;
            instance->lastMouseY = y;
        } else {
            instance->mousePressed = false;
        }
    }
}

void Renderer::mouseMotion(int x, int y) {
    if (!instance || !instance->mousePressed) return;

    int dx = x - instance->lastMouseX;
    int dy = y - instance->lastMouseY;

    instance->cameraAngleY += dx * 0.5f;
    instance->cameraAngleX += dy * 0.5f;

    // Clamp vertical angle
    if (instance->cameraAngleX > 89.0f) instance->cameraAngleX = 89.0f;
    if (instance->cameraAngleX < -89.0f) instance->cameraAngleX = -89.0f;

    instance->lastMouseX = x;
    instance->lastMouseY = y;

    glutPostRedisplay();
}

void Renderer::keyboard(unsigned char key, int x, int y) {
    if (!instance) return;

    switch (key) {
        case 27: // ESC
            std::cout << "Exiting..." << std::endl;
            exit(0);
            break;
        case ' ': // SPACE
            instance->showImages = !instance->showImages;
            if (instance->showImages) {
                std::cout << "Switched to MNIST images mode - Each sphere shows the digit that neuron learned" << std::endl;
            } else {
                std::cout << "Switched to color mode - Each color represents a digit class (0-9)" << std::endl;
            }
            glutPostRedisplay();
            break;
        case '+':
        case '=':
            instance->cameraDistance -= 0.5f;
            if (instance->cameraDistance < 2.0f) instance->cameraDistance = 2.0f;
            glutPostRedisplay();
        break;
        case '-':
            instance->cameraDistance += 0.5f;
            if (instance->cameraDistance > 20.0f) instance->cameraDistance = 20.0f;
            glutPostRedisplay();
        break;
    }
}

void Renderer::specialKeys(int key, int x, int y) {
    if (!instance) return;

    switch (key) {
        case GLUT_KEY_UP:
            instance->cameraAngleX += 5.0f;
            if (instance->cameraAngleX > 89.0f) instance->cameraAngleX = 89.0f;
            break;
        case GLUT_KEY_DOWN:
            instance->cameraAngleX -= 5.0f;
            if (instance->cameraAngleX < -89.0f) instance->cameraAngleX = -89.0f;
            break;
        case GLUT_KEY_LEFT:
            instance->cameraAngleY -= 5.0f;
            break;
        case GLUT_KEY_RIGHT:
            instance->cameraAngleY += 5.0f;
            break;
    }
    glutPostRedisplay();
}
