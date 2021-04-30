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

    const char *openblas_dirs[3];
    openblas_dirs[0] = "\\extra-dll\\";
    openblas_dirs[1] = "\\..\\numpy\\.libs\\";
    openblas_dirs[2] = "\\.libs\\";
    const int ndirs = sizeof(openblas_dirs)/sizeof(openblas_dirs[0]);
    const char *openblas_glob = "libopenblas*dll";

    const char *other_libs[2];
    other_libs[0] = "liblapack.dll";
    other_libs[1] = "mkl_rt.1.dll";
    const int nlibs = sizeof(other_libs)/sizeof(other_libs[0]);

    char buf[300];
    char err_info[2000] = "";

    for (i = 0; i < ndirs; i++) {
        snprintf(buf, sizeof(buf), "%s%s%s", scipy_dir, openblas_dirs[i], openblas_glob);
        snprintf(err_info + strlen(err_info), sizeof(err_info), "\nSearched for %s", buf);
        hfile = FindFirstFile(buf, &filedata);
        if (hfile != INVALID_HANDLE_VALUE) {
            break;
        }
    }

    if (hfile != INVALID_HANDLE_VALUE) {
        snprintf(err_info + strlen(err_info), sizeof(err_info),
                 "\nTried to load %s", filedata.cFileName);
        lib = LoadLibrary(filedata.cFileName);
    } else {
        // Try to load other possible libs, they should be on the path
        // so don't need the full path
        for (i = 0; i < nlibs; i++) {
            lib = LoadLibrary(other_libs[i]);
            snprintf(err_info + strlen(err_info), sizeof(err_info),
                     "\nTried to load %s", other_libs[i]);
            if (lib != NULL) {
                break;
            }
        }
    }

    if (lib == NULL) {
        printf(err_info);
        printf("\nCould not load lib handle");
        return NULL;
    }
    zheevd = (ZheevdFunc) GetProcAddress(lib, "zheevd_");
    if (zheevd == NULL) {
        printf(err_info);
        printf("\nCould not find zheevd_ in lib");
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
