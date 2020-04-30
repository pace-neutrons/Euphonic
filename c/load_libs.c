#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <glob.h>
#endif

#include <stdio.h>

typedef void (*ZheevdFunc)(char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, double* rwork, int* lrwork,
    int* iwork, int* liwork, int* info);

ZheevdFunc get_zheevd(const char *scipy_dir) {
    ZheevdFunc zheevd;

#ifdef _WIN32
    HMODULE lib;
    WIN32_FIND_DATA filedata;
    HANDLE hfile;
    int i;
    const char *libdirs[2];
    libdirs[0] = "\\extra-dll\\";
    libdirs[1] = "\\..\\numpy\\.libs\\";
    const int ndirs = sizeof(libdirs)/sizeof(libdirs[0]);
    const char *fileglob = "libopenblas*dll";
    char buf[300];
    for (i = 0; i < ndirs; i++) {
        snprintf(buf, sizeof(buf), "%s%s%s", scipy_dir, libdirs[i], fileglob);
        hfile = FindFirstFile(buf, &filedata);
        if (hfile != INVALID_HANDLE_VALUE) {
            break;
        }
        // libopenblas.dll file should have been found by this point, if not
        // print all searched dirs and return null
        if (i == ndirs - 1) {
            for (i = 0; i < ndirs; i++) {
                printf("Could not find %s\n", buf);
            }
            return NULL;
        }
    }

    lib = LoadLibrary(filedata.cFileName);
    if (lib == NULL) {
        printf("Could not load lib handle %s\n", filedata.cFileName);
        return NULL;
    }
    zheevd = (ZheevdFunc) GetProcAddress(lib, "zheevd_");
    if (zheevd == NULL) {
        printf("Could not find zheevd_ in %s\n", filedata.cFileName);
        FreeLibrary(lib);
        return NULL;
    }
    FreeLibrary(lib);
#else
    void *lib;
    const char *libdir = "/linalg";
    const char *fileglob = "/_flapack*so";
    glob_t globres;
    char buf[300];
    DIR *dir;

    snprintf(buf, sizeof(buf), "%s%s", scipy_dir, libdir);
    dir = opendir(buf);
    if (dir == NULL) {
        printf("Could not open dir %s\n", buf);
        return NULL;
    }
    while (readdir(dir) != NULL) {
        snprintf(buf, sizeof(buf), "%s%s%s", scipy_dir, libdir, fileglob);
        glob(buf, 0, NULL, &globres);
        if (globres.gl_pathc > 0) {
            break;
        }
    }
    closedir(dir);
    if (globres.gl_pathc == 0) {
        printf("Glob failed: couldn't find %s\n", buf);
        return NULL;
    }

    lib = dlopen(globres.gl_pathv[0], RTLD_LAZY);
    if (lib == NULL) {
        printf("Could not load lib handle %s. Error: %s\n", globres.gl_pathv[0], dlerror());
        return NULL;
    }
    zheevd = dlsym(lib, "zheevd_");
    if (zheevd == NULL) {
        printf("Could not find zheevd_ in %s\n", buf);
        dlclose(lib);
        return NULL;
    }
    dlclose(lib);
#endif

    return zheevd;
}
