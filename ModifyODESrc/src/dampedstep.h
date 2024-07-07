#pragma once
/*************************************************************************
* this file is added by Libin Liu
* for stepping the world with damped joints
*************************************************************************/

#ifndef _ODE_DAMPED_STEP_H_
#define _ODE_DAMPED_STEP_H_

#include <ode/common.h>

class dxWorldProcessMemArena;


size_t dxEstimateDamppedStepMemoryRequirements (
    dxBody * const *body, unsigned int nb, dxJoint * const *_joint, unsigned int _nj);

void dInternalDamppedStepIsland (dxWorldProcessMemArena *memarena, dxWorld *world,
    dxBody * const *body, unsigned int nb,
    dxJoint * const *joint, unsigned int nj,
    dReal stepsize);


#endif