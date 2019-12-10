#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <glob.h>
#endif

#include <stdio.h>
#include "lib_funcs.h"

ZheevdFunc get_zheevd(const char *scipy_dir) {
    ZheevdFunc zheevd;

#ifdef _WIN32
    HMODULE lib;
    WIN32_FIND_DATA filedata;
    HANDLE hfile;
    const char *libdir = "\\extra-dll\\";
    const char *fileglob = "libopenblas*dll";
    char buf[300];
    snprintf(buf, sizeof(buf), "%s%s%s", scipy_dir, libdir, fileglob);
    hfile = FindFirstFile(buf, &filedata);
    if (hfile == INVALID_HANDLE_VALUE) {
        printf("Could not find %s\n", buf);
        return NULL;
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

    snprintf(buf, sizeof(buf), "%s/%s", buf, globres.gl_pathv[0]);
    lib = dlopen(buf, RTLD_LAZY);
    if (lib == NULL) {
        printf("Could not load lib handle %s\n", buf);
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