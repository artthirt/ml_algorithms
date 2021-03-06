#ifndef QT_WORK_MAT_H
#define QT_WORK_MAT_H

#ifdef QT
#include <QObject>

#include "custom_types.h"
#include "gpumat.h"

namespace qt_work_mat{

/**
 * @brief q_save_mat
 * @param mat
 * @param filename
 */
void q_save_mat(const ct::Matf &mat, const QString &filename);
/**
 * @brief q_save_mat
 * @param mat
 * @param filename
 */
void q_save_mat(const gpumat::GpuMat &mat, const QString &filename);
/**
 * @brief q_load_mat
 * @param filename
 * @param mat
 */
void q_load_mat(const QString &filename, ct::Matf& mat);
/**
 * @brief q_load_mat
 * @param filename
 * @param mat
 */
void q_load_mat(const QString &filename, gpumat::GpuMat &mat);

}

#endif

#endif // QT_WORK_MAT_H
