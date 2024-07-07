from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import builtins as __builtins__
import copy as copy
import datetime as datetime
import enum
from enum import IntEnum
import json as json
import logging as logging
import numpy as np
import numpy
from numpy import uint64 as np_size_t
import os as os
import pickle as pickle
import platform as platform
import random as random
from scipy.interpolate._interpolate import interp1d
from scipy.ndimage import filters
from scipy import signal
from scipy.spatial.transform._rotation import Rotation
from scipy.spatial.transform._rotation import Rotation as R
import time as time
import tkinter as tk
from tkinter import filedialog
import typing
from typing import IO
import weakref as weakref
__all__ = ['AMotor', 'AMotorEuler', 'AMotorUser', 'ArgumentParser', 'BallJoint', 'BallJointAmotor', 'BallJointBase', 'Body', 'CloseODE', 'Common', 'Contact', 'ContactApprox1', 'ContactJoint', 'ContactJointBase', 'ContactJointMaxForce', 'ContactSlip1', 'ContactSlip2', 'DRigidBodyInertiaMode', 'DRigidBodyMassMode', 'EmptyBallJoint', 'FixedJoint', 'GeomBox', 'GeomCCylinder', 'GeomCapsule', 'GeomCylinder', 'GeomObject', 'GeomPlane', 'GeomRay', 'GeomSphere', 'GeomTransform', 'GeomTriMesh', 'GeomType', 'GeomTypes', 'HashSpace', 'Hinge2Joint', 'HingeJoint', 'IO', 'Inertia', 'Infinity', 'InitODE', 'IntEnum', 'Joint', 'JointGroup', 'JointParam', 'JointTypes', 'LMotor', 'Mass', 'Namespace', 'ODESim', 'PRJoint', 'Plane2DJoint', 'PyEigenArrayXXd', 'PyEigenArrayXd', 'PyEigenMatrixXd', 'PyEigenVectorXd', 'QuadTreeSpace', 'R', 'Render', 'Rotation', 'SetInitSeed', 'SimpleSpace', 'SliderJoint', 'Space', 'SpaceBase', 'TriMeshData', 'UniversalJoint', 'Utils', 'World', 'areConnected', 'collide', 'collide2', 'copy', 'datetime', 'debug_song', 'decompose_rotation_fast', 'decompose_rotation_pair_one2many_fast', 'decompose_rotation_single_fast', 'decompose_rotation_single_pair_fast', 'environment', 'filedialog', 'filters', 'flip_quat_by_w_fast', 'interp1d', 'json', 'logging', 'mix_quat_by_slerp', 'np', 'np_size_t', 'os', 'parent_child_quat_to_hinge_angle_fast', 'pickle', 'platform', 'pymotionlib', 'quat_apply_forward_fast', 'quat_apply_forward_one2many_fast', 'quat_apply_single_fast', 'quat_from_matrix_fast', 'quat_from_matrix_single_fast', 'quat_from_rotvec_fast', 'quat_from_rotvec_single_fast', 'quat_integrate_fast', 'quat_inv_fast', 'quat_inv_single_fast', 'quat_multiply_forward_fast', 'quat_multiply_forward_one2many_fast', 'quat_multiply_forward_single', 'quat_to_hinge_angle_fast', 'quat_to_matrix_fast', 'quat_to_matrix_single_fast', 'quat_to_rotvec_fast', 'quat_to_rotvec_fast2', 'quat_to_rotvec_single_fast', 'quat_to_vec6d_fast', 'quat_to_vec6d_single_fast', 'random', 'signal', 'simple_mix_quaternion', 'six_dim_mat_to_quat_fast', 'solve_lcp', 'surface_distance_capsule', 'test_py_list_str_to_std_vector_str', 'time', 'tk', 'vector_normalize_single_fast', 'vector_to_cross_matrix_fast', 'visDrawBackground', 'visDrawWorld', 'visEndRecordVideo', 'visGetScreenBuffer', 'visGetWorld', 'visKill', 'visLookAt', 'visPause', 'visSetAxisLength', 'visSetColor', 'visSetJointRadius', 'visSetWorld', 'visStartRecordVideo', 'visTrackBody', 'visWhetherHingeAxis', 'visWhetherLocalAxis', 'weakref']
class AMotor(Joint):
    """
    AMotor(World world, JointGroup jointgroup=None)
    AMotor joint.
    
        Constructor::AMotor(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        AMotor.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        AMotor.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addTorques(*args, **kwargs):
        """
        AMotor.addTorques(self, dReal torque0, dReal torque1, dReal torque2)
        addTorques(torque0, torque1, torque2)
        
                Applies torques about the AMotor's axes.
        
                @param torque0: Torque 0 magnitude
                @param torque1: Torque 1 magnitude
                @param torque2: Torque 2 magnitude
                @type torque0: float
                @type torque1: float
                @type torque2: float
                
        """
    @staticmethod
    def getAngle(*args, **kwargs):
        """
        AMotor.getAngle(self, int anum)
        getAngle(anum) -> float
        
                Return the current angle for axis anum.
        
                @param anum: Axis index
                @type anum: int
                
        """
    @staticmethod
    def getAngleRate(*args, **kwargs):
        """
        AMotor.getAngleRate(self, int anum)
        getAngleRate(anum) -> float
        
                Return the current angle rate for axis anum.
        
                @param anum: Axis index
                @type anum: int
                
        """
    @staticmethod
    def getAxis(*args, **kwargs):
        """
        AMotor.getAxis(self, int anum)
        getAxis(anum)
        
                Get an AMotor axis.
        
                @param anum: Axis index (0-2)
                @type anum: int
                
        """
    @staticmethod
    def getAxisNumpy(*args, **kwargs):
        """
        AMotor.getAxisNumpy(self, int anum)
        """
    @staticmethod
    def getAxisRel(*args, **kwargs):
        """
        AMotor.getAxisRel(self, int anum)
        getAxisRel(anum) -> int
        
                Get the relative mode of an axis.
        
                @param anum: Axis index (0-2)
                @type anum: int
                
        """
    @staticmethod
    def getMode(*args, **kwargs):
        """
        AMotor.getMode(self)
        getMode()
        
                Return the angular motor mode (AMotorUser or AMotorEuler).
                
        """
    @staticmethod
    def getNumAxes(*args, **kwargs):
        """
        AMotor.getNumAxes(self)
        getNumAxes() -> int
        
                Get the number of angular axes that are controlled by the AMotor.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        AMotor.getParam(self, int param)
        """
    @staticmethod
    def setAngle(*args, **kwargs):
        """
        AMotor.setAngle(self, int anum, dReal angle)
        setAngle(anum, angle)
        
                Tell the AMotor what the current angle is along axis anum.
        
                @param anum: Axis index
                @param angle: Angle
                @type anum: int
                @type angle: float
                
        """
    @staticmethod
    def setAxis(*args, **kwargs):
        """
        AMotor.setAxis(self, int anum, int rel, axis)
        setAxis(anum, rel, axis)
        
                Set an AMotor axis.
        
                The anum argument selects the axis to change (0,1 or 2).
                Each axis can have one of three "relative orientation" modes,
                selected by rel:
        
                0: The axis is anchored to the global frame.
                1: The axis is anchored to the first body.
                2: The axis is anchored to the second body.
        
                The axis vector is always specified in global coordinates
                regardless of the setting of rel.
        
                @param anum: Axis number
                @param rel: Relative orientation mode
                @param axis: Axis
                @type anum: int
                @type rel: int
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setAxisNumpy(*args, **kwargs):
        """
        AMotor.setAxisNumpy(self, int anum, int rel, ndarray np_axis)
        """
    @staticmethod
    def setMode(*args, **kwargs):
        """
        AMotor.setMode(self, int mode)
        setMode(mode)
        
                Set the angular motor m  mode must be either AMotorUser or
                AMotorEuler.
        
                @param mode: Angular motor mode
                @type mode: int
                
        """
    @staticmethod
    def setNumAxes(*args, **kwargs):
        """
        AMotor.setNumAxes(self, int num)
        setNumAxes(num)
        
                Set the number of angular axes that will be controlled by the AMotor.
                num may be in the range from 0 to 3.
        
                @param num: Number of axes (0-3)
                @type num: int
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        AMotor.setParam(self, int param, dReal value)
        """
class BallJoint(BallJointBase):
    """
    BallJoint(World world, JointGroup jointgroup=None)
    Ball joint.
    
        Constructor::BallJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        BallJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        BallJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        BallJoint.getAnchor(self)
        getAnchor() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates.  This
                returns the point on body 1.  If the joint is perfectly
                satisfied, this will be the same as the point on body 2.
                
        """
    @staticmethod
    def getAnchor1Raw(*args, **kwargs):
        """
        BallJoint.getAnchor1Raw(self)
        """
    @staticmethod
    def getAnchor1RawNumpy(*args, **kwargs):
        """
        BallJoint.getAnchor1RawNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        BallJoint.getAnchor2(self)
        getAnchor2() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates.  This
                returns the point on body 2. If the joint is perfectly
                satisfied, this will be the same as the point on body 1.
                
        """
    @staticmethod
    def getAnchor2Numpy(*args, **kwargs):
        """
        BallJoint.getAnchor2Numpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchor2Raw(*args, **kwargs):
        """
        BallJoint.getAnchor2Raw(self)
        """
    @staticmethod
    def getAnchor2RawNumpy(*args, **kwargs):
        """
        BallJoint.getAnchor2RawNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchorNumpy(*args, **kwargs):
        """
        BallJoint.getAnchorNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        BallJoint.getParam(self, int param)
        """
    @staticmethod
    def get_joint_dof(*args, **kwargs):
        """
        BallJoint.get_joint_dof(self) -> int
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        BallJoint.setAnchor(self, pos)
        setAnchor(pos)
        
                Set the joint anchor point which must be specified in world
                coordinates.
        
                @param pos: Anchor position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setAnchorNumpy(*args, **kwargs):
        """
        BallJoint.setAnchorNumpy(self, ndarray np_pos)
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        BallJoint.setParam(self, int param, dReal value)
        """
class BallJointAmotor(BallJointBase):
    """
    BallJointAmotor(World world, JointGroup jointgroup=None)
    Ball joint with AMotor.
    
        Constructor::BallJointAmotor(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        BallJointAmotor.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        BallJointAmotor.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _destroy_amotor(*args, **kwargs):
        """
        BallJointAmotor._destroy_amotor(self)
        """
    @staticmethod
    def _setData(*args, **kwargs):
        """
        BallJointAmotor._setData(self, value)
        """
    @staticmethod
    def addAmotorTorques(*args, **kwargs):
        """
        BallJointAmotor.addAmotorTorques(self, dReal torque0, dReal torque1, dReal torque2)
        """
    @staticmethod
    def attach_ext(*args, **kwargs):
        """
        BallJointAmotor.attach_ext(self, Body body1, Body body2)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        BallJointAmotor.destroy_immediate(self)
        """
    @staticmethod
    def getAmotorAllAxisNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAllAxisNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAmotorAngle(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAngle(self, int anum)
        """
    @staticmethod
    def getAmotorAngleRate(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAngleRate(self, int anum)
        """
    @staticmethod
    def getAmotorAngleRateNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAngleRateNumpy(self)
        """
    @staticmethod
    def getAmotorAxis(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAxis(self, int anum)
        """
    @staticmethod
    def getAmotorAxisNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAxisNumpy(self, int anum) -> np.ndarray
        """
    @staticmethod
    def getAmotorAxisRel(*args, **kwargs):
        """
        BallJointAmotor.getAmotorAxisRel(self, int anum)
        """
    @staticmethod
    def getAmotorMode(*args, **kwargs):
        """
        BallJointAmotor.getAmotorMode(self)
        """
    @staticmethod
    def getAmotorParam(*args, **kwargs):
        """
        BallJointAmotor.getAmotorParam(self, int param)
        """
    @staticmethod
    def getAmtorNumAxes(*args, **kwargs):
        """
        BallJointAmotor.getAmtorNumAxes(self)
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        BallJointAmotor.getAnchor(self)
        """
    @staticmethod
    def getAnchor1Raw(*args, **kwargs):
        """
        BallJointAmotor.getAnchor1Raw(self)
        """
    @staticmethod
    def getAnchor1RawNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAnchor1RawNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        BallJointAmotor.getAnchor2(self)
        """
    @staticmethod
    def getAnchor2Numpy(*args, **kwargs):
        """
        BallJointAmotor.getAnchor2Numpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchor2Raw(*args, **kwargs):
        """
        BallJointAmotor.getAnchor2Raw(self)
        """
    @staticmethod
    def getAnchor2RawNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAnchor2RawNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAnchorNumpy(*args, **kwargs):
        """
        BallJointAmotor.getAnchorNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAngleLimit1(*args, **kwargs):
        """
        BallJointAmotor.getAngleLimit1(self)
        """
    @staticmethod
    def getAngleLimit2(*args, **kwargs):
        """
        BallJointAmotor.getAngleLimit2(self)
        """
    @staticmethod
    def getAngleLimit3(*args, **kwargs):
        """
        BallJointAmotor.getAngleLimit3(self)
        """
    @staticmethod
    def get_amotor_jid(*args, **kwargs):
        """
        BallJointAmotor.get_amotor_jid(self)
        """
    @staticmethod
    def get_joint_dof(*args, **kwargs):
        """
        BallJointAmotor.get_joint_dof(self)
        """
    @staticmethod
    def setAmotorAngle(*args, **kwargs):
        """
        BallJointAmotor.setAmotorAngle(self, int anum, dReal angle)
        """
    @staticmethod
    def setAmotorAxis(*args, **kwargs):
        """
        BallJointAmotor.setAmotorAxis(self, int anum, int rel, axis)
        """
    @staticmethod
    def setAmotorAxisNumpy(*args, **kwargs):
        """
        BallJointAmotor.setAmotorAxisNumpy(self, int anum, int rel, ndarray np_axis)
        """
    @staticmethod
    def setAmotorMode(*args, **kwargs):
        """
        BallJointAmotor.setAmotorMode(self, int mode)
        """
    @staticmethod
    def setAmotorNumAxes(*args, **kwargs):
        """
        BallJointAmotor.setAmotorNumAxes(self, int num)
        """
    @staticmethod
    def setAmotorParam(*args, **kwargs):
        """
        BallJointAmotor.setAmotorParam(self, int param, dReal value)
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        BallJointAmotor.setAnchor(self, pos)
        """
    @staticmethod
    def setAnchorNumpy(*args, **kwargs):
        """
        BallJointAmotor.setAnchorNumpy(self, ndarray np_pos)
        """
    @staticmethod
    def setAngleLim1(*args, **kwargs):
        """
        BallJointAmotor.setAngleLim1(self, dReal lo, dReal hi)
        """
    @staticmethod
    def setAngleLim2(*args, **kwargs):
        """
        BallJointAmotor.setAngleLim2(self, dReal lo, dReal hi)
        """
    @staticmethod
    def setAngleLim3(*args, **kwargs):
        """
        BallJointAmotor.setAngleLim3(self, dReal lo, dReal hi)
        """
class BallJointBase(Joint):
    """
    BallJointBase(World world, JointGroup jointgroup=None)
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        BallJointBase.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        BallJointBase.__setstate_cython__(self, __pyx_state)
        """
class Body:
    """
    Body(World world)
    The rigid body class encapsulating the ODE body.
    
        This class represents a rigid body that has a location and orientation
        in space and that stores the mass properties of an object.
    
        When creating a Body object you have to pass the world it belongs to
        as argument to the constructor::
    
        >>> import ode
        >>> w = World()
        >>> b = Body(w)
        
    """
    __hash__: typing.ClassVar[None] = None
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Constructor.
        
                @param world: The world in which the body should be created.
                @type world: World
                
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Body.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Body.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _getBodyData(*args, **kwargs):
        """
        Body._getBodyData(self)
        """
    @staticmethod
    def _getData(*args, **kwargs):
        """
        Body._getData(self)
        """
    @staticmethod
    def _setData(*args, **kwargs):
        """
        Body._setData(self, data)
        """
    @staticmethod
    def addForce(*args, **kwargs):
        """
        Body.addForce(self, f)
        addForce(f)
        
                Add an external force f given in absolute coordinates. The force
                is applied at the center of mass.
        
                @param f: Force
                @type f: 3-sequence of floats
                
        """
    @staticmethod
    def addForceAtPos(*args, **kwargs):
        """
        Body.addForceAtPos(self, f, p)
        addForceAtPos(f, p)
        
                Add an external force f at position p. Both arguments must be
                given in absolute coordinates.
        
                @param f: Force
                @param p: Position
                @type f: 3-sequence of floats
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def addForceAtPosNumpy(*args, **kwargs):
        """
        Body.addForceAtPosNumpy(self, ndarray f, ndarray p)
        
                param: f: np.ndarray in shape (3,)
                p: np.ndarray in shape (3,)
                
        """
    @staticmethod
    def addForceAtRelPos(*args, **kwargs):
        """
        Body.addForceAtRelPos(self, f, p)
        addForceAtRelPos(f, p)
        
                Add an external force f at position p. f is given in absolute
                coordinates and p in absolute coordinates.
        
                @param f: Force
                @param p: Position
                @type f: 3-sequence of floats
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def addForceNumpy(*args, **kwargs):
        """
        Body.addForceNumpy(self, ndarray f)
        """
    @staticmethod
    def addRelForce(*args, **kwargs):
        """
        Body.addRelForce(self, f)
        addRelForce(f)
        
                Add an external force f given in relative coordinates
                (relative to the body's own frame of reference). The force
                is applied at the center of mass.
        
                @param f: Force
                @type f: 3-sequence of floats
                
        """
    @staticmethod
    def addRelForceAtPos(*args, **kwargs):
        """
        Body.addRelForceAtPos(self, f, p)
        addRelForceAtPos(f, p)
        
                Add an external force f at position p. f is given in relative
                coordinates and p in relative coordinates.
        
                @param f: Force
                @param p: Position
                @type f: 3-sequence of floats
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def addRelForceAtRelPos(*args, **kwargs):
        """
        Body.addRelForceAtRelPos(self, f, p)
        addRelForceAtRelPos(f, p)
        
                Add an external force f at position p. Both arguments must be
                given in relative coordinates.
        
                @param f: Force
                @param p: Position
                @type f: 3-sequence of floats
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def addRelTorque(*args, **kwargs):
        """
        Body.addRelTorque(self, t)
        addRelTorque(t)
        
                Add an external torque t given in relative coordinates
                (relative to the body's own frame of reference).
        
                @param t: Torque
                @type t: 3-sequence of floats
                
        """
    @staticmethod
    def addTorque(*args, **kwargs):
        """
        Body.addTorque(self, t)
        addTorque(t)
        
                Add an external torque t given in absolute coordinates.
        
                @param t: Torque
                @type t: 3-sequence of floats
                
        """
    @staticmethod
    def addTorqueNumpy(*args, **kwargs):
        """
        Body.addTorqueNumpy(self, ndarray t)
        """
    @staticmethod
    def copy_body(*args, **kwargs):
        """
        Body.copy_body(self, SpaceBase space=None)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        Body.destroy_immediate(self)
        """
    @staticmethod
    def disable(*args, **kwargs):
        """
        Body.disable(self)
        disable()
        
                Manually disable a body. Note that a disabled body that is connected
                through a joint to an enabled body will be automatically re-enabled
                at the next simulation step.
                
        """
    @staticmethod
    def enable(*args, **kwargs):
        """
        Body.enable(self)
        enable()
        
                Manually enable a body.
                
        """
    @staticmethod
    def geom_iter(*args, **kwargs):
        """
        Body.geom_iter(self)
        """
    @staticmethod
    def getAngularVel(*args, **kwargs):
        """
        Body.getAngularVel(self)
        getAngularVel() -> 3-tuple
        
                Get the current angular velocity of the body.
                
        """
    @staticmethod
    def getAngularVelNumpy(*args, **kwargs):
        """
        Body.getAngularVelNumpy(self)
        """
    @staticmethod
    def getFacingInfo(*args, **kwargs):
        """
        Body.getFacingInfo(self, Body root_body=None, int is_vec6d=1)
        """
    @staticmethod
    def getFacingQuaternion(*args, **kwargs):
        """
        Body.getFacingQuaternion(self)
        
                get the decomposed rotation (remove y rotation component)
                
        """
    @staticmethod
    def getFacingRotVec(*args, **kwargs):
        """
        Body.getFacingRotVec(self)
        """
    @staticmethod
    def getFacingVec6d(*args, **kwargs):
        """
        Body.getFacingVec6d(self)
        """
    @staticmethod
    def getFiniteRotationAxis(*args, **kwargs):
        """
        Body.getFiniteRotationAxis(self)
        getFiniteRotationAxis() -> 3-tuple
        
                Return the current finite rotation axis of the body.
                
        """
    @staticmethod
    def getFiniteRotationMode(*args, **kwargs):
        """
        Body.getFiniteRotationMode(self) -> int
        getFiniteRotationMode() -> mode (0/1)
        
                Return the current finite rotation mode of a body (0 or 1).
                See setFiniteRotationMode().
                
        """
    @staticmethod
    def getForce(*args, **kwargs):
        """
        Body.getForce(self)
        getForce() -> 3-tuple
        
                Return the current accumulated force.
                
        """
    @staticmethod
    def getForceNumpy(*args, **kwargs):
        """
        Body.getForceNumpy(self)
        """
    @staticmethod
    def getGeomIDNumpy(*args, **kwargs):
        """
        Body.getGeomIDNumpy(self)
        """
    @staticmethod
    def getGravityMode(*args, **kwargs):
        """
        Body.getGravityMode(self) -> int
        getGravityMode() -> bool
        
                Return True if the body is influenced by the world's gravity.
                
        """
    @staticmethod
    def getJoint(*args, **kwargs):
        """
        Body.getJoint(self, int idx)
        """
    @staticmethod
    def getLinearVel(*args, **kwargs):
        """
        Body.getLinearVel(self)
        getLinearVel() -> 3-tuple
        
                Get the current linear velocity of the body.
                
        """
    @staticmethod
    def getMass(*args, **kwargs):
        """
        Body.getMass(self) -> Mass
        getMass() -> mass
        
                Return the mass properties as a Mass object.
                
        """
    @staticmethod
    def getNumGeom(*args, **kwargs):
        """
        Body.getNumGeom(self) -> int
        """
    @staticmethod
    def getNumJoints(*args, **kwargs):
        """
        Body.getNumJoints(self)
        getNumJoints() -> int
        
                Return the number of joints that are attached to this body.
                
        """
    @staticmethod
    def getPointVel(*args, **kwargs):
        """
        Body.getPointVel(self, p)
        getPointVel(p) -> 3-tuple
        
                Utility function that takes a point p on a body and returns
                that point's velocity in global coordinates. The point p
                must be given in global coordinates.
        
                @param p: Body point (global coordinates)
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def getPosRelPoint(*args, **kwargs):
        """
        Body.getPosRelPoint(self, p)
        getPosRelPoint(p) -> 3-tuple
        
                This is the inverse of getRelPointPos(). It takes a point p in
                global coordinates and returns the point's position in
                body-relative coordinates.
        
                @param p: Body point (global coordinates)
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def getPosRelPointNumpy(*args, **kwargs):
        """
        Body.getPosRelPointNumpy(self, ndarray pos)
        
                It takes a point p in global coordinates
                and returns the point's position in body-relative coordinates.
        
                @param p: Body point (global coordinates)
                
        """
    @staticmethod
    def getQuaternionScipy(*args, **kwargs):
        """
        Body.getQuaternionScipy(self)
        """
    @staticmethod
    def getRelPointPos(*args, **kwargs):
        """
        Body.getRelPointPos(self, p)
        getRelPointPos(p) -> 3-tuple
        
                Utility function that takes a point p on a body and returns
                that point's position in global coordinates. The point p
                must be given in body relative coordinates.
        
                @param p: Body point (local coordinates)
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def getRelPointVel(*args, **kwargs):
        """
        Body.getRelPointVel(self, p)
        getRelPointVel(p) -> 3-tuple
        
                Utility function that takes a point p on a body and returns
                that point's velocity in global coordinates. The point p
                must be given in body relative coordinates.
        
                @param p: Body point (local coordinates)
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def getRotation(*args, **kwargs):
        """
        Body.getRotation(self)
        getRotation() -> 9-tuple
        
                Return the current rotation matrix as a tuple of 9 floats (row-major
                order).
                
        """
    @staticmethod
    def getRotationAxisAngle(*args, **kwargs):
        """
        Body.getRotationAxisAngle(self)
        """
    @staticmethod
    def getRotationNumpy(*args, **kwargs):
        """
        Body.getRotationNumpy(self)
        """
    @staticmethod
    def getRotationVec6d(*args, **kwargs):
        """
        Body.getRotationVec6d(self)
        """
    @staticmethod
    def getTorque(*args, **kwargs):
        """
        Body.getTorque(self)
        getTorque() -> 3-tuple
        
                Return the current accumulated torque.
                
        """
    @staticmethod
    def getTorqueNumpy(*args, **kwargs):
        """
        Body.getTorqueNumpy(self)
        """
    @staticmethod
    def get_aabb(*args, **kwargs):
        """
        Body.get_aabb(self) -> np.ndarray
        """
    @staticmethod
    def get_bid(*args, **kwargs):
        """
        Body.get_bid(self) -> size_t
        """
    @staticmethod
    def isKinematic(*args, **kwargs):
        """
        Body.isKinematic(self) -> int
        isKinematic() -> bool
        
                Return True if the body is kinematic (not influenced by other forces).
        
                Kinematic bodies behave as if they had infinite mass. This means they don't react
                to any force (gravity, constraints or user-supplied); they simply follow
                velocity to reach the next position. [from ODE wiki]
        
                
        """
    @staticmethod
    def setAngularVel(*args, **kwargs):
        """
        Body.setAngularVel(self, vel)
        setAngularVel(vel)
        
                Set the angular velocity of the body.
        
                @param vel: New angular velocity
                @type vel: 3-sequence of floats
                
        """
    @staticmethod
    def setAngularVelNumpy(*args, **kwargs):
        """
        Body.setAngularVelNumpy(self, ndarray ang_vel)
        """
    @staticmethod
    def setDynamic(*args, **kwargs):
        """
        Body.setDynamic(self)
        setDynamic()
        
                Set a body to the (default) "dynamic" state, instead of "kinematic".
                See setKinematic() for more information.
                
        """
    @staticmethod
    def setFiniteRotationAxis(*args, **kwargs):
        """
        Body.setFiniteRotationAxis(self, a)
        setFiniteRotationAxis(a)
        
                Set the finite rotation axis of the body.  This axis only has a
                meaning when the finite rotation mode is set
                (see setFiniteRotationMode()).
        
                @param a: Axis
                @type a: 3-sequence of floats
                
        """
    @staticmethod
    def setFiniteRotationMode(*args, **kwargs):
        """
        Body.setFiniteRotationMode(self, int mode)
        setFiniteRotationMode(mode)
        
                This function controls the way a body's orientation is updated at
                each time step. The mode argument can be:
        
                - 0: An "infinitesimal" orientation update is used. This is
                fast to compute, but it can occasionally cause inaccuracies
                for bodies that are rotating at high speed, especially when
                those bodies are joined to other bodies. This is the default
                for every new body that is created.
        
                - 1: A "finite" orientation update is used. This is more
                costly to compute, but will be more accurate for high speed
                rotations. Note however that high speed rotations can result
                in many types of error in a simulation, and this mode will
                only fix one of those sources of error.
        
                @param mode: Rotation mode (0/1)
                @type mode: int
                
        """
    @staticmethod
    def setForce(*args, **kwargs):
        """
        Body.setForce(self, f)
        setForce(f)
        
                Set the body force accumulation vector.
        
                @param f: Force
                @type f: 3-tuple of floats
                
        """
    @staticmethod
    def setForceNumpy(*args, **kwargs):
        """
        Body.setForceNumpy(self, ndarray f)
        """
    @staticmethod
    def setGravityMode(*args, **kwargs):
        """
        Body.setGravityMode(self, mode)
        setGravityMode(mode)
        
                Set whether the body is influenced by the world's gravity
                or not. If mode is True it is, otherwise it isn't.
                Newly created bodies are always influenced by the world's gravity.
        
                @param mode: Gravity mode
                @type mode: bool
                
        """
    @staticmethod
    def setKinematic(*args, **kwargs):
        """
        Body.setKinematic(self)
        setKinematic()
        
                Set the kinematic state of the body (change it into a kinematic body)
        
                Kinematic bodies behave as if they had infinite mass. This means they don't react
                to any force (gravity, constraints or user-supplied); they simply follow 
                velocity to reach the next position. [from ODE wiki]
        
                
        """
    @staticmethod
    def setLinearVel(*args, **kwargs):
        """
        Body.setLinearVel(self, vel)
        setLinearVel(vel)
        
                Set the linear velocity of the body.
        
                @param vel: New velocity
                @type vel: 3-sequence of floats
                
        """
    @staticmethod
    def setMass(*args, **kwargs):
        """
        Body.setMass(self, Mass mass)
        setMass(mass)
        
                Set the mass properties of the body. The argument mass must be
                an instance of a Mass object.
        
                @param mass: Mass properties
                @type mass: Mass
                
        """
    @staticmethod
    def setMaxAngularSpeed(*args, **kwargs):
        """
        Body.setMaxAngularSpeed(self, dReal max_speed)
        setMaxAngularSpeed(max_speed)
        
                You can also limit the maximum angular speed. In contrast to the damping
                functions, the angular velocity is affected before the body is moved.
                This means that it will introduce errors in joints that are forcing the
                body to rotate too fast. Some bodies have naturally high angular
                velocities (like cars' wheels), so you may want to give them a very high
                (like the default, dInfinity) limit.
        
                
        """
    @staticmethod
    def setQuaternionScipy(*args, **kwargs):
        """
        Body.setQuaternionScipy(self, ndarray q)
        """
    @staticmethod
    def setRotAndQuatNoNormScipy(*args, **kwargs):
        """
        Body.setRotAndQuatNoNormScipy(self, ndarray Rot, ndarray quat)
        """
    @staticmethod
    def setRotationNumpy(*args, **kwargs):
        """
        Body.setRotationNumpy(self, ndarray Rot)
        setRotationNumpy(Rot)
        
                Set the orientation of the body. The rotation matrix must be
                given as a sequence of 9 floats which are the elements of the
                matrix in row-major order.
        
                @param R: Rotation matrix
                @type R: 9-sequence of floats
                
        """
    @staticmethod
    def setTorque(*args, **kwargs):
        """
        Body.setTorque(self, t)
        setTorque(t)
        
                Set the body torque accumulation vector.
        
                @param t: Torque
                @type t: 3-tuple of floats
                
        """
    @staticmethod
    def setTorqueNumpy(*args, **kwargs):
        """
        Body.setTorqueNumpy(self, ndarray t)
        """
    @staticmethod
    def set_draw_local_axis(*args, **kwargs):
        """
        Body.set_draw_local_axis(self, x)
        """
    @staticmethod
    def vectorFromWorld(*args, **kwargs):
        """
        Body.vectorFromWorld(self, v)
        vectorFromWorld(v) -> 3-tuple
        
                Given a vector v expressed in the world coordinate system, rotate
                it to the body coordinate system.
        
                @param v: Vector in world coordinate system
                @type v: 3-sequence of floats
                
        """
    @staticmethod
    def vectorToWorld(*args, **kwargs):
        """
        Body.vectorToWorld(self, v)
        vectorToWorld(v) -> 3-tuple
        
                Given a vector v expressed in the body coordinate system, rotate
                it to the world coordinate system.
        
                @param v: Vector in body coordinate system
                @type v: 3-sequence of floats
                
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
class Common:
    class GetFileNameByUI:
        @staticmethod
        def get_file_name_by_UI(initialdir = './', filetypes = [('all_file_types', '*.*')]):
            """
            GetFileNameByUI.get_file_name_by_UI(initialdir=u'./', filetypes=[(u'all_file_types', u'*.*')])
            """
        @staticmethod
        def get_multi_file_name_by_UI():
            """
            GetFileNameByUI.get_multi_file_name_by_UI()
            """
    class Helper:
        _empty_str_list: typing.ClassVar[list] = ['', 'null', 'none', 'nullptr', 'no', 'not', 'false', 'abort']
        _true_str_list: typing.ClassVar[list] = ['yes', 'true', 'confirm', 'ok', 'sure', 'ready']
        @staticmethod
        def conf_loader(fname: unicode) -> typing.Dict[str, typing.Any]:
            """
            Helper.conf_loader(unicode fname: str) -> Dict[str, Any]
            """
        @staticmethod
        def get_curr_time() -> unicode:
            """
            Helper.get_curr_time() -> str
            """
        @staticmethod
        def is_str_empty(s: unicode) -> bool:
            """
            Helper.is_str_empty(unicode s: str) -> bool
            """
        @staticmethod
        def load_numpy_random_state(result: typing.Dict[str, typing.Any]) -> None:
            """
            Helper.load_numpy_random_state(result: Dict[str, Any]) -> None
            """
        @staticmethod
        def mirror_name_list(name_list: typing.List[str]):
            """
            Helper.mirror_name_list(name_list: List[str])
            """
        @staticmethod
        def print_total_time(starttime):
            """
            Helper.print_total_time(starttime)
            """
        @staticmethod
        def save_numpy_random_state() -> typing.Dict[str, typing.Any]:
            """
            Helper.save_numpy_random_state() -> Dict[str, Any]
            """
        @staticmethod
        def set_torch_seed(random_seed: int):
            """
            Helper.set_torch_seed(random_seed: int)
            """
        @staticmethod
        def str_is_true(s: unicode) -> bool:
            """
            Helper.str_is_true(unicode s: str) -> bool
            """
        def __init__(self):
            """
            Helper.__init__(self)
            """
    class MathHelper:
        unit_vec6d: typing.ClassVar[numpy.ndarray]  # value = array([1., 0., 0., 1., 0., 0.])
        @staticmethod
        def average_quat_by_slerp(qs: typing.List[numpy.ndarray]) -> ndarray:
            """
            MathHelper.average_quat_by_slerp(qs: List[np.ndarray]) -> np.ndarray
            """
        @staticmethod
        def axis_decompose(q: ndarray, axis: ndarray):
            """
            MathHelper.axis_decompose(ndarray q: np.ndarray, ndarray axis: np.ndarray)
            
                        return:
                        res: rotation along axis
                        r_other:
                        
            """
        @staticmethod
        def count_ones(x: int) -> int:
            """
            MathHelper.count_ones(x: int) -> int
            """
        @staticmethod
        def cross_mat(v):
            """
            MathHelper.cross_mat(v)
            create cross-product matrix for v
            
                        Args:
                            v (torch.Tensor): a vector with shape (..., 3, 1)
                        
            """
        @staticmethod
        def decompose_rotation(q: ndarray, vb: ndarray):
            """
            MathHelper.decompose_rotation(ndarray q: np.ndarray, ndarray vb: np.ndarray)
            """
        @staticmethod
        def ego_forward_vector() -> ndarray:
            """
            MathHelper.ego_forward_vector() -> np.ndarray
            
                        return (0, 0, 1)
                        
            """
        @staticmethod
        def facing_decompose(q: ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
            """
            MathHelper.facing_decompose(ndarray q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            
                        return: ry, facing. ry only has y component, and facing only has (x, z) component.
                        
            """
        @staticmethod
        def flip_quat_arr_by_dot(*args) -> typing.List[numpy.ndarray]:
            """
            MathHelper.flip_quat_arr_by_dot(*args) -> List[np.ndarray]
            """
        @staticmethod
        def flip_quat_arr_by_w(*args):
            """
            MathHelper.flip_quat_arr_by_w(*args)
            """
        @staticmethod
        def flip_quat_by_dot(q: ndarray, inplace: bool = False) -> ndarray:
            """
            MathHelper.flip_quat_by_dot(ndarray q: np.ndarray, inplace: bool = False) -> np.ndarray
            """
        @staticmethod
        def flip_quat_by_w(q: ndarray) -> ndarray:
            """
            MathHelper.flip_quat_by_w(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def flip_quat_pair_by_dot(q0s: ndarray, q1s: ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
            """
            MathHelper.flip_quat_pair_by_dot(ndarray q0s: np.ndarray, ndarray q1s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            
                        q0 will not be changed.
                        q1 will be flipped to the same semi sphere as q0
                        
            """
        @staticmethod
        def flip_vec3_by_dot(x: ndarray, inplace: bool = False) -> ndarray:
            """
            MathHelper.flip_vec3_by_dot(ndarray x: np.ndarray, inplace: bool = False) -> np.ndarray
            """
        @staticmethod
        def flip_vector_by_dot(x: ndarray, inplace: bool = False) -> ndarray:
            """
            MathHelper.flip_vector_by_dot(ndarray x: np.ndarray, inplace: bool = False) -> np.ndarray
            
                        make sure x[i] * x[i+1] >= 0
                        
            """
        @staticmethod
        def get_rotation_dim(rotate_type):
            """
            MathHelper.get_rotation_dim(rotate_type)
            """
        @staticmethod
        def get_rotation_last_shape(rotate_type) -> typing.Tuple:
            """
            MathHelper.get_rotation_last_shape(rotate_type) -> Tuple
            """
        @staticmethod
        def normalize_angle(a: ndarray) -> ndarray:
            """
            MathHelper.normalize_angle(ndarray a: np.ndarray) -> np.ndarray
            
                        Covert angles to [-pi, pi)
                        
            """
        @staticmethod
        def normalize_vec(a: ndarray) -> ndarray:
            """
            MathHelper.normalize_vec(ndarray a: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def np_skew(v: ndarray):
            """
            MathHelper.np_skew(ndarray v: np.ndarray)
            """
        @staticmethod
        def ode_quat_to_rot_mat(q: ndarray) -> ndarray:
            """
            MathHelper.ode_quat_to_rot_mat(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def proj_multi_vec_to_a_plane(a_arr: ndarray, v: ndarray):
            """
            MathHelper.proj_multi_vec_to_a_plane(ndarray a_arr: np.ndarray, ndarray v: np.ndarray)
            """
        @staticmethod
        def quat_between(a: ndarray, b: ndarray) -> ndarray:
            """
            MathHelper.quat_between(ndarray a: np.ndarray, ndarray b: np.ndarray) -> np.ndarray
            
                        Rotation from vector a to vector b
                        :param a: (n, 3) vector
                        :param b: (n, 3) vector
                        :return: (n, 4) quaternion
                        
            """
        @staticmethod
        def quat_equal(q1: ndarray, q2: ndarray) -> bool:
            """
            MathHelper.quat_equal(ndarray q1: np.ndarray, ndarray q2: np.ndarray) -> bool
            """
        @staticmethod
        def quat_from_ode_to_scipy(q: ndarray) -> ndarray:
            """
            MathHelper.quat_from_ode_to_scipy(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def quat_from_ode_to_unity(q: ndarray) -> ndarray:
            """
            MathHelper.quat_from_ode_to_unity(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def quat_from_other_rotate(x: ndarray, rotate_type) -> ndarray:
            """
            MathHelper.quat_from_other_rotate(ndarray x: np.ndarray, rotate_type) -> np.ndarray
            """
        @staticmethod
        def quat_from_scipy_to_ode(q: ndarray) -> ndarray:
            """
            MathHelper.quat_from_scipy_to_ode(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def quat_to_other_rotate(quat: ndarray, rotate_type):
            """
            MathHelper.quat_to_other_rotate(ndarray quat: np.ndarray, rotate_type)
            """
        @staticmethod
        def rot_mat_to_ode_quat(mat: ndarray) -> ndarray:
            """
            MathHelper.rot_mat_to_ode_quat(ndarray mat: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def torch_skew(v):
            """
            MathHelper.torch_skew(v)
            
                        :param v : torch.Tensor [3,1] or [1,3]
                        this function will return the skew matrix (cross product matrix) of a vector
                        be sure that it has ONLY 3 element
                        it can be autograd
                        
            """
        @staticmethod
        def unit_quat() -> ndarray:
            """
            MathHelper.unit_quat() -> np.ndarray
            """
        @staticmethod
        def unit_quat_arr(shape: typing.Union[int, typing.Iterable, typing.Tuple[int]]) -> ndarray:
            """
            MathHelper.unit_quat_arr(shape: Union[int, Iterable, Tuple[int]]) -> np.ndarray
            """
        @staticmethod
        def unit_quat_ode() -> ndarray:
            """
            MathHelper.unit_quat_ode() -> np.ndarray
            """
        @staticmethod
        def unit_quat_ode_list() -> typing.List[float]:
            """
            MathHelper.unit_quat_ode_list() -> List[float]
            """
        @staticmethod
        def unit_quat_scipy() -> ndarray:
            """
            MathHelper.unit_quat_scipy() -> np.ndarray
            """
        @staticmethod
        def unit_quat_scipy_list() -> typing.List[float]:
            """
            MathHelper.unit_quat_scipy_list() -> List[float]
            """
        @staticmethod
        def unit_quat_unity() -> ndarray:
            """
            MathHelper.unit_quat_unity() -> np.ndarray
            """
        @staticmethod
        def unit_quat_unity_list() -> typing.List[float]:
            """
            MathHelper.unit_quat_unity_list() -> List[float]
            """
        @staticmethod
        def unit_vector(axis: int) -> ndarray:
            """
            MathHelper.unit_vector(axis: int) -> np.ndarray
            """
        @staticmethod
        def up_vector() -> ndarray:
            """
            MathHelper.up_vector() -> np.ndarray
            
                        return (0, 1, 0)
                        
            """
        @staticmethod
        def vec_axis_to_zero(v: ndarray, axis: typing.Union[int, typing.List[int], numpy.ndarray]) -> ndarray:
            """
            MathHelper.vec_axis_to_zero(ndarray v: np.ndarray, axis: Union[int, List[int], np.ndarray]) -> np.ndarray
            """
        @staticmethod
        def vec_diff(v_in: ndarray, forward: bool, fps: float):
            """
            MathHelper.vec_diff(ndarray v_in: np.ndarray, forward: bool, double fps: float)
            """
        @staticmethod
        def wxyz_to_xyzw(q: ndarray) -> ndarray:
            """
            MathHelper.wxyz_to_xyzw(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def x_decompose(q: ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
            """
            MathHelper.x_decompose(ndarray q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            """
        @staticmethod
        def xyzw_to_wxyz(q: ndarray) -> ndarray:
            """
            MathHelper.xyzw_to_wxyz(ndarray q: np.ndarray) -> np.ndarray
            """
        @staticmethod
        def y_decompose(q: ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
            """
            MathHelper.y_decompose(ndarray q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            """
        @staticmethod
        def z_decompose(q: ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
            """
            MathHelper.z_decompose(ndarray q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            """
    class RotateConvertFast:
        @staticmethod
        def quat_from_other_rotate(x: ndarray, rotate_type) -> ndarray:
            """
            RotateConvertFast.quat_from_other_rotate(ndarray x: np.ndarray, rotate_type) -> np.ndarray
            """
        @staticmethod
        def quat_single_from_other_rotate(x: ndarray, rotate_type) -> ndarray:
            """
            RotateConvertFast.quat_single_from_other_rotate(ndarray x: np.ndarray, rotate_type) -> np.ndarray
            """
        @staticmethod
        def quat_single_to_other_rotate(x: ndarray, rotate_type) -> ndarray:
            """
            RotateConvertFast.quat_single_to_other_rotate(ndarray x: np.ndarray, rotate_type) -> np.ndarray
            """
        @staticmethod
        def quat_to_other_rotate(x: ndarray, rotate_type) -> ndarray:
            """
            RotateConvertFast.quat_to_other_rotate(ndarray x: np.ndarray, rotate_type) -> np.ndarray
            """
    class RotateType(enum.IntEnum):
        """
        An enumeration.
        """
        AxisAngle: typing.ClassVar[Common.RotateType]  # value = <RotateType.AxisAngle: 2>
        Matrix: typing.ClassVar[Common.RotateType]  # value = <RotateType.Matrix: 1>
        Quaternion: typing.ClassVar[Common.RotateType]  # value = <RotateType.Quaternion: 5>
        SVD9d: typing.ClassVar[Common.RotateType]  # value = <RotateType.SVD9d: 4>
        Vec6d: typing.ClassVar[Common.RotateType]  # value = <RotateType.Vec6d: 3>
    class SmoothOperator:
        class ButterWorthBase(Common.SmoothOperator.FilterInfoBase):
            __slots__: typing.ClassVar[tuple] = ('order', 'wn')
            def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
                """
                ButterWorthBase.__init__(self, order: int, double cut_off_freq: float, sample_freq: int)
                """
        class FilterInfoBase:
            __slots__: typing.ClassVar[tuple] = ('order', 'wn')
            @staticmethod
            def calc_freq(cut_off_freq: float, sample_freq: float) -> float:
                """
                FilterInfoBase.calc_freq(double cut_off_freq: float, double sample_freq: float) -> float
                """
            @classmethod
            def build_from_dict(cls, info: typing.Optional[typing.Dict[str, typing.Any]], sample_freq: int):
                """
                FilterInfoBase.build_from_dict(cls, info: Optional[Dict[str, Any]], sample_freq: int)
                """
            def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
                """
                FilterInfoBase.__init__(self, order: int, double cut_off_freq: float, sample_freq: int)
                """
        class GaussianBase:
            __slots__: typing.ClassVar[tuple] = ('width')
            def __init__(self, width: typing.Optional[int]):
                """
                GaussianBase.__init__(self, width: Optional[int])
                """
        class SmoothMode(enum.IntEnum):
            """
            An enumeration.
            """
            BUTTER_WORTH: typing.ClassVar[Common.SmoothOperator.SmoothMode]  # value = <SmoothMode.BUTTER_WORTH: 2>
            GAUSSIAN: typing.ClassVar[Common.SmoothOperator.SmoothMode]  # value = <SmoothMode.GAUSSIAN: 1>
            NO: typing.ClassVar[Common.SmoothOperator.SmoothMode]  # value = <SmoothMode.NO: 0>
        @staticmethod
        def smooth_operator(x: ndarray, smooth_type) -> ndarray:
            """
            SmoothOperator.smooth_operator(ndarray x: np.ndarray, smooth_type) -> np.ndarray
            
                        The first dimension of x is time
                        
            """
class Contact:
    """
    This class represents a contact between two bodies in one point.
    
        A Contact object stores all the input parameters for a ContactJoint.
        This class wraps the ODE dContact structure which has 3 components::
    
        struct dContact {
            dSurfaceParameters surface;
            dContactGeom geom;
            dVector3 fdir1;
        };
    
        This wrapper class provides methods to get and set the items of those
        structures.
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Contact.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Contact.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def enable_contact_slip(*args, **kwargs):
        """
        Contact.enable_contact_slip(self, dReal slip)
        """
    @staticmethod
    def enable_soft_cfm_erp(*args, **kwargs):
        """
        Contact.enable_soft_cfm_erp(self, dReal cfm, dReal erp)
        """
    @staticmethod
    def getContactGeomParams(*args, **kwargs):
        """
        Contact.getContactGeomParams(self)
        getContactGeomParams() -> (pos, normal, depth, geom1, geom2)
        
                Get the ContactGeom structure of the contact.
        
                The return value is a tuple (pos, normal, depth, geom1, geom2)
                where pos and normal are 3-tuples of floats and depth is a single
                float. geom1 and geom2 are the Geom objects of the geoms in contact.
                
        """
    @staticmethod
    def setContactGeomParams(*args, **kwargs):
        """
        Contact.setContactGeomParams(self, ndarray pos, ndarray normal, dReal depth, GeomObject g1=None, GeomObject g2=None)
        setContactGeomParams(pos, normal, depth, geom1=None, geom2=None)
        
                Set the ContactGeom structure of the contact.
                
        """
class ContactJoint(ContactJointBase):
    """
    ContactJoint(World world, JointGroup jointgroup, Contact contact)
    Contact joint.
    
        Constructor::ContactJoint(world, jointgroup, contact)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        ContactJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        ContactJoint.__setstate_cython__(self, __pyx_state)
        """
class ContactJointBase(Joint):
    """
    ContactJointBase(*a, **kw)
    
        base class for all types of contact joint
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        ContactJointBase.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        ContactJointBase.__setstate_cython__(self, __pyx_state)
        """
class ContactJointMaxForce(ContactJointBase):
    """
    ContactJointMaxForce(World world, JointGroup jointgroup, Contact contact)
    
        A simplified contact model.
    
        simple, the formula is simplified as follow:
        0 <= support force <= +infty
        friction 0 <= contact mu (or max friction)
        friction 1 <= contact mu (or max friction)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        ContactJointMaxForce.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        ContactJointMaxForce.__setstate_cython__(self, __pyx_state)
        """
class DRigidBodyInertiaMode(enum.IntEnum):
    """
    
        Compute the inertia of rigid body by density or given inertia value
        
    """
    Density: typing.ClassVar[DRigidBodyInertiaMode]  # value = <DRigidBodyInertiaMode.Density: 0>
    InertiaValue: typing.ClassVar[DRigidBodyInertiaMode]  # value = <DRigidBodyInertiaMode.InertiaValue: 1>
class DRigidBodyMassMode(enum.IntEnum):
    """
    
            Compute the mass of rigid body by density or given mass value
            
    """
    Density: typing.ClassVar[DRigidBodyMassMode]  # value = <DRigidBodyMassMode.Density: 0>
    MassValue: typing.ClassVar[DRigidBodyMassMode]  # value = <DRigidBodyMassMode.MassValue: 1>
class EmptyBallJoint(Joint):
    """
    EmptyBallJoint(World world)
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        EmptyBallJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        EmptyBallJoint.__setstate_cython__(self, __pyx_state)
        """
class FixedJoint(Joint):
    """
    FixedJoint(World world, JointGroup jointgroup=None)
    Fixed joint.
    
        Constructor::FixedJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        FixedJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        FixedJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def setFixed(*args, **kwargs):
        """
        FixedJoint.setFixed(self)
        setFixed()
        
                Call this on the fixed joint after it has been attached to
                remember the current desired relative offset and desired
                relative rotation between the bodies.
                
        """
class GeomBox(GeomObject):
    """
    GeomBox(SpaceBase space=None, lengths=(1.0, 1.0, 1.0))
    Box geometry.
    
        This class represents a box centered at the origin.
    
        Constructor::GeomBox(space=None, lengths=(1.0, 1.0, 1.0))
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomBox.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomBox.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getLengthNumpy(*args, **kwargs):
        """
        GeomBox.getLengthNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getLengths(*args, **kwargs):
        """
        GeomBox.getLengths(self)
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomBox.placeable(self) -> bool
        """
    @staticmethod
    def pointDepth(*args, **kwargs):
        """
        GeomBox.pointDepth(self, p)
        pointDepth(p) -> float
        
                Return the depth of the point p in the box. Points inside the
                geom will have positive depth, points outside it will have
                negative depth, and points on the surface will have zero
                depth.
        
                @param p: Point
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def setLengths(*args, **kwargs):
        """
        GeomBox.setLengths(self, lengths)
        """
class GeomCapsule(GeomObject):
    """
    GeomCapsule(SpaceBase space=None, dReal radius=0.5, dReal length=1.0)
    Capped cylinder geometry.
    
        This class represents a capped cylinder aligned along the local Z axis
        and centered at the origin.
    
        Constructor::
        GeomCapsule(space=None, radius=0.5, length=1.0)
    
        The length parameter does not include the caps.
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomCapsule.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomCapsule.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def capsule_axis(*args, **kwargs):
        """
        GeomCapsule.capsule_axis(self)
        
                
        """
    @staticmethod
    def getParams(*args, **kwargs):
        """
        GeomCapsule.getParams(self)
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomCapsule.placeable(self)
        """
    @staticmethod
    def pointDepth(*args, **kwargs):
        """
        GeomCapsule.pointDepth(self, p)
        pointDepth(p) -> float
        
                Return the depth of the point p in the cylinder. Points inside the
                geom will have positive depth, points outside it will have
                negative depth, and points on the surface will have zero
                depth.
        
                @param p: Point
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def setParams(*args, **kwargs):
        """
        GeomCapsule.setParams(self, dReal radius, dReal length)
        """
class GeomCylinder(GeomObject):
    """
    GeomCylinder(SpaceBase space=None, dReal radius=0.5, dReal length=1.0)
    Plain cylinder geometry.
    
        Note: ou should compile ode with libccd for supporting collision detection between cylinder and other geoms.
    
        This class represents an uncapped cylinder aligned along the local Z axis
        and centered at the origin.
    
        Constructor:: GeomCylinder(space=None, radius=0.5, length=1.0)
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomCylinder.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomCylinder.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getParams(*args, **kwargs):
        """
        GeomCylinder.getParams(self)
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomCylinder.placeable(self)
        """
    @staticmethod
    def setParams(*args, **kwargs):
        """
        GeomCylinder.setParams(self, dReal radius, dReal length)
        """
class GeomObject:
    """
    GeomObject(*a, **kw)
    This is the abstract base class for all geom objects.
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomObject.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomObject.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _getData(*args, **kwargs):
        """
        GeomObject._getData(self)
        """
    @staticmethod
    def _setData(*args, **kwargs):
        """
        GeomObject._setData(self, value)
        """
    @staticmethod
    def append_ignore_geom(*args, **kwargs):
        """
        GeomObject.append_ignore_geom(self, GeomObject other)
        """
    @staticmethod
    def clearOffset(*args, **kwargs):
        """
        GeomObject.clearOffset(self)
        clearOffset()
        
                Disable the offset transform of the geom.
                
        """
    @staticmethod
    def copy_geom(*args, **kwargs):
        """
        GeomObject.copy_geom(self, Body body, SpaceBase space)
        """
    @staticmethod
    def copy_pos_quat(*args, **kwargs):
        """
        GeomObject.copy_pos_quat(self, GeomObject result)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        GeomObject.destroy_immediate(self)
        """
    @staticmethod
    def disable(*args, **kwargs):
        """
        GeomObject.disable(self)
        disable()
        
                Disable the geom.
        """
    @staticmethod
    def enable(*args, **kwargs):
        """
        GeomObject.enable(self)
        enable()
        
                Enable the geom.
        """
    @staticmethod
    def extend_ignore_geom_id(*args, **kwargs):
        """
        GeomObject.extend_ignore_geom_id(self, list res)
        """
    @staticmethod
    def getClass(*args, **kwargs):
        """
        GeomObject.getClass(self) -> int
        """
    @staticmethod
    def getOffsetPosition(*args, **kwargs):
        """
        GeomObject.getOffsetPosition(self)
        getOffsetPosition() -> 3-tuple
        
                Get the offset position of the geom.
                
        """
    @staticmethod
    def getOffsetPositionNumpy(*args, **kwargs):
        """
        GeomObject.getOffsetPositionNumpy(self)
        """
    @staticmethod
    def getOffsetRotation(*args, **kwargs):
        """
        GeomObject.getOffsetRotation(self)
        getOffsetRotation() -> 9-tuple
        
                Get the offset rotation of the geom.
                
        """
    @staticmethod
    def getOffsetRotationNumpy(*args, **kwargs):
        """
        GeomObject.getOffsetRotationNumpy(self) -> np.ndarray
        """
    @staticmethod
    def get_gid(*args, **kwargs):
        """
        GeomObject.get_gid(self) -> size_t
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomObject.placeable(self) -> bool
        placeable() -> bool
        
                Returns True if the geom object is a placeable geom.
        
                This method has to be overwritten in derived methods.
                
        """
    @staticmethod
    def setOffsetPosition(*args, **kwargs):
        """
        GeomObject.setOffsetPosition(self, pos)
        setOffsetPosition(pos)
        
                Set the offset position of the geom. The geom must be attached to a
                body.  If the geom did not have an offset, it is automatically created.
                This sets up an additional (local) transformation for the geom, since
                geoms attached to a body share their global position and rotation.
        
                @param pos: Position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setOffsetRotationNumpy(*args, **kwargs):
        """
        GeomObject.setOffsetRotationNumpy(self, ndarray rot)
        setOffsetRotationNumpy(R)
        
                Set the offset rotation of the geom. The geom must be attached to a
                body.  If the geom did not have an offset, it is automatically created.
                This sets up an additional (local) transformation for the geom, since
                geoms attached to a body share their global position and rotation.
        
                @param R: Rotation matrix
                @type R: 9-sequence of floats
                
        """
    @staticmethod
    def setOffsetWorldPositionNumpy(*args, **kwargs):
        """
        GeomObject.setOffsetWorldPositionNumpy(self, ndarray pos)
        """
    @staticmethod
    def setOffsetWorldRotationNumpy(*args, **kwargs):
        """
        GeomObject.setOffsetWorldRotationNumpy(self, ndarray rot)
        """
    @staticmethod
    def set_draw_local_axis(*args, **kwargs):
        """
        GeomObject.set_draw_local_axis(self, x)
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __hash__(self):
        """
        Return hash(self).
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
class GeomPlane(GeomObject):
    """
    GeomPlane(SpaceBase space=None, normal=(0, 0, 1), dist=0)
    Plane geometry.
    
        This class represents an infinite plane. The plane equation is:
        n.x*x + n.y*y + n.z*z = dist
    
        This object can't be attached to a body.
        If you call getBody() on this object it always returns environment.
    
        Constructor::GeomPlane(space=None, normal=(0,0,1), dist=0)
    
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomPlane.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomPlane.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getParams(*args, **kwargs):
        """
        GeomPlane.getParams(self)
        """
    @staticmethod
    def pointDepth(*args, **kwargs):
        """
        GeomPlane.pointDepth(self, p)
        pointDepth(p) -> float
        
                Return the depth of the point p in the plane. Points inside the
                geom will have positive depth, points outside it will have
                negative depth, and points on the surface will have zero
                depth.
        
                @param p: Point
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def setParams(*args, **kwargs):
        """
        GeomPlane.setParams(self, normal, dist)
        """
class GeomRay(GeomObject):
    """
    GeomRay(SpaceBase space=None, dReal rlen=1.0)
    Ray object.
    
        A ray is different from all the other geom classes in that it does
        not represent a solid object. It is an infinitely thin line that
        starts from the geom's position and extends in the direction of
        the geom's local Z-axis.
    
        Constructor:: GeomRay(space=None, rlen=1.0)
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomRay.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomRay.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def get(*args, **kwargs):
        """
        GeomRay.get(self)
        get() -> ((p[0], p[1], p[2]), (u[0], u[1], u[2]))
        
                Return the position and rotation as a pair of
                tuples.
        
                @returns: position and rotation
        """
    @staticmethod
    def getLength(*args, **kwargs):
        """
        GeomRay.getLength(self)
        getLength() -> length
        
                Get the length of the ray.
        
                @returns: length of the ray (float)
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomRay.placeable(self)
        """
    @staticmethod
    def set(*args, **kwargs):
        """
        GeomRay.set(self, p, u)
        set(p, u)
        
                Set the position and rotation of a ray.
        
                @param p: position
                @type p: 3-sequence of floats
                @param u: rotation
                @type u: 3-sequence of floats
        """
    @staticmethod
    def setLength(*args, **kwargs):
        """
        GeomRay.setLength(self, rlen)
        setLength(rlen)
        
                Set length of the ray.
        
                @param rlen: length of the ray
                @type rlen: float
        """
class GeomSphere(GeomObject):
    """
    GeomSphere(SpaceBase space=None, dReal radius=1.0)
    Sphere geometry.
    
        This class represents a sphere centered at the origin.
    
        Constructor::GeomSphere(space=None, radius=1.0)
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomSphere.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomSphere.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def copy_geom(*args, **kwargs):
        """
        GeomSphere.copy_geom(self, Body body, SpaceBase space)
        """
    @staticmethod
    def getRadius(*args, **kwargs):
        """
        GeomSphere.getRadius(self)
        getRadius() -> float
        
                Return the radius of the sphere.
                
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomSphere.placeable(self)
        """
    @staticmethod
    def pointDepth(*args, **kwargs):
        """
        GeomSphere.pointDepth(self, p)
        pointDepth(p) -> float
        
                Return the depth of the point p in the sphere. Points inside
                the geom will have positive depth, points outside it will have
                negative depth, and points on the surface will have zero
                depth.
        
                @param p: Point
                @type p: 3-sequence of floats
                
        """
    @staticmethod
    def setRadius(*args, **kwargs):
        """
        GeomSphere.setRadius(self, dReal radius)
        setRadius(radius)
        
                Set the radius of the sphere.
        
                @param radius: New radius
                @type radius: float
                
        """
class GeomTransform(GeomObject):
    """
    GeomTransform(SpaceBase space=None)
    GeomTransform.
    
        A geometry transform "T" is a geom that encapsulates another geom
        "E", allowing E to be positioned and rotated arbitrarily with
        respect to its point of reference.
    
        Constructor::GeomTransform(space=None)
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomTransform.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomTransform.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getGeom(*args, **kwargs):
        """
        GeomTransform.getGeom(self)
        getGeom() -> GeomObject
        
                Get the geom that the geometry transform encapsulates.
                
        """
    @staticmethod
    def getInfo(*args, **kwargs):
        """
        GeomTransform.getInfo(self)
        getInfo() -> int
        
                Get the "information" mode of the geometry transform (0 or 1).
        
                With mode 0, when a transform object is collided with another
                object, the geom field of the ContactGeom structure is set to the
                geom that is encapsulated by the transform object.
        
                With mode 1, the geom field of the ContactGeom structure is set
                to the transform object itself.
                
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomTransform.placeable(self)
        """
    @staticmethod
    def setGeom(*args, **kwargs):
        """
        GeomTransform.setGeom(self, GeomObject geom)
        setGeom(geom)
        
                Set the geom that the geometry transform encapsulates.
                A ValueError exception is thrown if a) the geom is not placeable,
                b) the geom was already inserted into a space or c) the geom is
                already associated with a body.
        
                @param geom: Geom object to encapsulate
                @type geom: GeomObject
                
        """
    @staticmethod
    def setInfo(*args, **kwargs):
        """
        GeomTransform.setInfo(self, int mode)
        setInfo(mode)
        
                Set the "information" mode of the geometry transform.
        
                With mode 0, when a transform object is collided with another
                object, the geom field of the ContactGeom structure is set to the
                geom that is encapsulated by the transform object.
        
                With mode 1, the geom field of the ContactGeom structure is set
                to the transform object itself.
        
                @param mode: Information mode (0 or 1)
                @type mode: int
                
        """
class GeomTriMesh(GeomObject):
    """
    GeomTriMesh(TriMeshData data, space=None)
    TriMesh object.
    
        To construct the trimesh geom you need a TriMeshData object that
        stores the actual mesh. This object has to be passed as first
        argument to the constructor.
    
        Constructor::
        
          GeomTriMesh(data, space=None)
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomTriMesh.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomTriMesh.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _id(*args, **kwargs):
        """
        GeomTriMesh._id(self)
        """
    @staticmethod
    def clearTCCache(*args, **kwargs):
        """
        GeomTriMesh.clearTCCache(self)
        clearTCCache()
        
                Clears the internal temporal coherence caches.
                
        """
    @staticmethod
    def getTriangle(*args, **kwargs):
        """
        GeomTriMesh.getTriangle(self, int idx)
        getTriangle(idx) -> (v0, v1, v2)
        
                @param idx: Triangle index
                @type idx: int
                
        """
    @staticmethod
    def getTriangleCount(*args, **kwargs):
        """
        GeomTriMesh.getTriangleCount(self)
        getTriangleCount() -> n
        
                Returns the number of triangles in the TriMesh.
        """
    @staticmethod
    def placeable(*args, **kwargs):
        """
        GeomTriMesh.placeable(self)
        """
class GeomType:
    """
    
        parse geometry type
        
    """
    def __init__(self):
        """
        GeomType.__init__(self)
        """
    def is_box(self, geom_type: unicode):
        """
        GeomType.is_box(self, unicode geom_type: str)
        """
    def is_capsule(self, geom_type: unicode):
        """
        GeomType.is_capsule(self, unicode geom_type: str)
        """
    def is_plane(self, geom_type: unicode):
        """
        GeomType.is_plane(self, unicode geom_type: str)
        """
    def is_sphere(self, geom_type: unicode):
        """
        GeomType.is_sphere(self, unicode geom_type: str)
        """
    @property
    def box_type(self) -> unicode:
        """
        GeomType.box_type(self) -> str
        """
    @property
    def capsule_type(self) -> unicode:
        """
        GeomType.capsule_type(self) -> str
        """
    @property
    def plane_type(self) -> unicode:
        """
        GeomType.plane_type(self) -> str
        """
    @property
    def sphere_type(self) -> unicode:
        """
        GeomType.sphere_type(self) -> str
        """
class GeomTypes:
    Box: typing.ClassVar[int] = 1
    Capsule: typing.ClassVar[int] = 2
    Cylinder: typing.ClassVar[int] = 3
    Plane: typing.ClassVar[int] = 4
    Sphere: typing.ClassVar[int] = 0
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        GeomTypes.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        GeomTypes.__setstate_cython__(self, __pyx_state)
        """
class HashSpace(SpaceBase):
    """
    HashSpace(SpaceBase space=None)
    Multi-resolution hash table space.
    
        This uses an internal data structure that records how each geom
        overlaps cells in one of several three dimensional grids. Each
        grid has cubical cells of side lengths 2**i, where i is an integer
        that ranges from a minimum to a maximum value. The time required
        to do intersection testing for n objects is O(n) (as long as those
        objects are not clustered together too closely), as each object
        can be quickly paired with the objects around it.
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        HashSpace.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        HashSpace.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getLevels(*args, **kwargs):
        """
        HashSpace.getLevels(self)
        getLevels() -> (minlevel, maxlevel)
        
                Gets the size of the smallest and largest cell used in the
                hash table. The actual size is 2^minlevel and 2^maxlevel
                respectively.
                
        """
    @staticmethod
    def setLevels(*args, **kwargs):
        """
        HashSpace.setLevels(self, int minlevel, int maxlevel)
        setLevels(minlevel, maxlevel)
        
                Sets the size of the smallest and largest cell used in the
                hash table. The actual size will be 2^minlevel and 2^maxlevel
                respectively.
                
        """
class Hinge2Joint(Joint):
    """
    Hinge2Joint(World world, JointGroup jointgroup=None)
    Hinge2 joint.
    
        Constructor::Hinge2Joint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Hinge2Joint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Hinge2Joint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addTorques(*args, **kwargs):
        """
        Hinge2Joint.addTorques(self, torque1, torque2)
        addTorques(torque1, torque2)
        
                Applies torque1 about axis 1, and torque2 about axis 2.
        
                @param torque1: Torque 1 magnitude
                @param torque2: Torque 2 magnitude
                @type torque1: float
                @type torque2: float
                
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        Hinge2Joint.getAnchor(self)
        getAnchor() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 1. If the joint is perfectly satisfied, this
                will be the same as the point on body 2.
                
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        Hinge2Joint.getAnchor2(self)
        getAnchor2() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 2. If the joint is perfectly satisfied, this
                will be the same as the point on body 1.
                
        """
    @staticmethod
    def getAngle1(*args, **kwargs):
        """
        Hinge2Joint.getAngle1(self) -> dReal
        getAngle1() -> float
        
                Get the first hinge-2 angle (around axis 1).
        
                When the anchor or axis is set, the current position of the
                attached bodies is examined and that position will be the zero
                angle.
                
        """
    @staticmethod
    def getAngle1Rate(*args, **kwargs):
        """
        Hinge2Joint.getAngle1Rate(self) -> dReal
        getAngle1Rate() -> float
        
                Get the time derivative of the first hinge-2 angle.
                
        """
    @staticmethod
    def getAngle2Rate(*args, **kwargs):
        """
        Hinge2Joint.getAngle2Rate(self) -> dReal
        getAngle2Rate() -> float
        
                Get the time derivative of the second hinge-2 angle.
                
        """
    @staticmethod
    def getAxis1(*args, **kwargs):
        """
        Hinge2Joint.getAxis1(self)
        getAxis1() -> 3-tuple of floats
        
                Get the first hinge-2 axis.
                
        """
    @staticmethod
    def getAxis2(*args, **kwargs):
        """
        Hinge2Joint.getAxis2(self)
        getAxis2() -> 3-tuple of floats
        
                Get the second hinge-2 axis.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        Hinge2Joint.getParam(self, int param)
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        Hinge2Joint.setAnchor(self, pos)
        setAnchor(pos)
        
                Set the hinge-2 anchor.
        
                @param pos: Anchor position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis1(*args, **kwargs):
        """
        Hinge2Joint.setAxis1(self, axis)
        setAxis1(axis)
        
                Set the first hinge-2 axis. Axis 1 and axis 2 must not lie
                along the same line.
        
                @param axis: Joint axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis2(*args, **kwargs):
        """
        Hinge2Joint.setAxis2(self, axis)
        setAxis2(axis)
        
                Set the second hinge-2 axis. Axis 1 and axis 2 must not lie
                along the same line.
        
                @param axis: Joint axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        Hinge2Joint.setParam(self, int param, dReal value)
        """
class HingeJoint(Joint):
    """
    HingeJoint(World world, JointGroup jointgroup=None)
    Hinge joint.
    
        Constructor::HingeJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        HingeJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        HingeJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addTorque(*args, **kwargs):
        """
        HingeJoint.addTorque(self, torque)
        addTorque(torque)
        
                Applies the torque about the hinge axis.
        
                @param torque: Torque magnitude
                @type torque: float
                
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        HingeJoint.getAnchor(self)
        getAnchor() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 1. If the joint is perfectly satisfied, this
                will be the same as the point on body 2.
                
        """
    @staticmethod
    def getAnchor1Raw(*args, **kwargs):
        """
        HingeJoint.getAnchor1Raw(self)
        """
    @staticmethod
    def getAnchor1RawNumpy(*args, **kwargs):
        """
        HingeJoint.getAnchor1RawNumpy(self)
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        HingeJoint.getAnchor2(self)
        getAnchor2() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 2. If the joint is perfectly satisfied, this
                will be the same as the point on body 1.
                
        """
    @staticmethod
    def getAnchor2Numpy(*args, **kwargs):
        """
        HingeJoint.getAnchor2Numpy(self)
        """
    @staticmethod
    def getAnchor2Raw(*args, **kwargs):
        """
        HingeJoint.getAnchor2Raw(self)
        """
    @staticmethod
    def getAnchor2RawNumpy(*args, **kwargs):
        """
        HingeJoint.getAnchor2RawNumpy(self)
        """
    @staticmethod
    def getAnchorNumpy(*args, **kwargs):
        """
        HingeJoint.getAnchorNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getAxis(*args, **kwargs):
        """
        HingeJoint.getAxis(self)
        getAxis() -> 3-tuple of floats
        
                Get the hinge axis.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        HingeJoint.getParam(self, int param)
        getParam(param) -> float
        
                Get limit/motor parameters for the joint.
        
                param is one of ParamLoStop, ParamHiStop, ParamVel, ParamFMax,
                ParamFudgeFactor, ParamBounce, ParamCFM, ParamStopERP, ParamStopCFM,
                ParamSuspensionERP, ParamSuspensionCFM.
        
                These parameter names can be optionally followed by a digit (2
                or 3) to indicate the second or third set of parameters.
        
                @param param: Selects the parameter to read
                @type param: int
                
        """
    @staticmethod
    def getQRelScipy(*args, **kwargs):
        """
        HingeJoint.getQRelScipy(self) -> np.ndarray
        """
    @staticmethod
    def get_joint_dof(*args, **kwargs):
        """
        HingeJoint.get_joint_dof(self)
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        HingeJoint.setAnchor(self, pos)
        setAnchor(pos)
        
                Set the hinge anchor which must be given in world coordinates.
        
                @param pos: Anchor position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setAnchorNumpy(*args, **kwargs):
        """
        HingeJoint.setAnchorNumpy(self, ndarray np_pos)
        """
    @staticmethod
    def setAngleLimit(*args, **kwargs):
        """
        HingeJoint.setAngleLimit(self, dReal lo, dReal hi)
        """
    @staticmethod
    def setAxis(*args, **kwargs):
        """
        HingeJoint.setAxis(self, axis)
        setAxis(axis)
        
                Set the hinge axis.
        
                @param axis: Hinge axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        HingeJoint.setParam(self, int param, dReal value)
        setParam(param, value)
        
                Set limit/motor parameters for the joint.
        
                param is one of ParamLoStop, ParamHiStop, ParamVel, ParamFMax,
                ParamFudgeFactor, ParamBounce, ParamCFM, ParamStopERP, ParamStopCFM,
                ParamSuspensionERP, ParamSuspensionCFM.
        
                These parameter names can be optionally followed by a digit (2
                or 3) to indicate the second or third set of parameters.
        
                @param param: Selects the parameter to set
                @param value: Parameter value
                @type param: int
                @type value: float
                
        """
class Inertia:
    @staticmethod
    def RotInertia(*args, **kwargs):
        """
        Inertia.RotInertia(self, ndarray np_rot)
        
                when a rigid body is rotated, the inertia is also modified
                
        """
    @staticmethod
    def TransInertia(*args, **kwargs):
        """
        Inertia.TransInertia(self, dReal tx, dReal ty, dReal tz)
        
                when a rigid body is translated by (tx, ty, tz), the inertia is also modified
                
        """
    @staticmethod
    def TransInertiaNumpy(*args, **kwargs):
        """
        Inertia.TransInertiaNumpy(self, ndarray t)
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Inertia.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Inertia.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def add(*args, **kwargs):
        """
        Inertia.add(self, Inertia o)
        """
    @staticmethod
    def getI(*args, **kwargs):
        """
        Inertia.getI(self)
        """
    @staticmethod
    def getMass(*args, **kwargs):
        """
        Inertia.getMass(self)
        """
    @staticmethod
    def setFromMassClass(*args, **kwargs):
        """
        Inertia.setFromMassClass(self, Mass m)
        """
    @staticmethod
    def setI(*args, **kwargs):
        """
        Inertia.setI(self, ndarray I)
        """
    @staticmethod
    def setMass(*args, **kwargs):
        """
        Inertia.setMass(self, dReal mass)
        
                set mass value
                
        """
    @staticmethod
    def setMassAndInertia(*args, **kwargs):
        """
        Inertia.setMassAndInertia(self, dReal mass, ndarray I)
        """
    @staticmethod
    def toMass(*args, **kwargs):
        """
        Inertia.toMass(self)
        
                convert to Mass object
                
        """
class Joint:
    """
    Joint(*a, **kw)
    Base class for all joint classes.
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def FeedBackForce(*args, **kwargs):
        """
        Joint.FeedBackForce(self) -> np.ndarray
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Joint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Joint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _destroyed(*args, **kwargs):
        """
        Joint._destroyed(self)
        Notify the joint object about an external destruction of the ODE joint.
        
                This method has to be called when the underlying ODE object
                was destroyed by someone else (e.g. by a joint group). The Python
                wrapper will then refrain from destroying it again.
                
        """
    @staticmethod
    def _getData(*args, **kwargs):
        """
        Joint._getData(self)
        """
    @staticmethod
    def _setData(*args, **kwargs):
        """
        Joint._setData(self, value)
        """
    @staticmethod
    def attach(*args, **kwargs):
        """
        Joint.attach(self, Body body1, Body body2)
        attach(body1, body2)
        
                Attach the joint to some new bodies. A body can be attached
                to the environment by passing None as second body.
        
                @param body1: First body
                @param body2: Second body
                @type body1: Body
                @type body2: Body
                
        """
    @staticmethod
    def attach_ext(*args, **kwargs):
        """
        Joint.attach_ext(self, Body body1, Body body2)
        """
    @staticmethod
    def dJointGetBody(*args, **kwargs):
        """
        Joint.dJointGetBody(self, int index)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        Joint.destroy_immediate(self)
        """
    @staticmethod
    def detach(*args, **kwargs):
        """
        Joint.detach(self)
        """
    @staticmethod
    def disable(*args, **kwargs):
        """
        Joint.disable(self)
        disable()
        
                Disable the joint. Disabled joints are completely ignored during the
                simulation. Disabled joints don't lose the already computed information
                like anchors and axes.
                
        """
    @staticmethod
    def disable_implicit_damping(*args, **kwargs):
        """
        Joint.disable_implicit_damping(self)
        """
    @staticmethod
    def enable(*args, **kwargs):
        """
        Joint.enable(self)
        enable()
        
                Enable the joint. Disabled joints are completely ignored during the
                simulation. Disabled joints don't lose the already computed information
                like anchors and axes.
                
        """
    @staticmethod
    def enable_implicit_damping(*args, **kwargs):
        """
        Joint.enable_implicit_damping(self)
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        Joint.getAnchor(self)
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        Joint.getAnchor2(self)
        """
    @staticmethod
    def getAnchor2Numpy(*args, **kwargs):
        """
        Joint.getAnchor2Numpy(self)
        """
    @staticmethod
    def getAnchorNumpy(*args, **kwargs):
        """
        Joint.getAnchorNumpy(self)
        """
    @staticmethod
    def getBody(*args, **kwargs):
        """
        Joint.getBody(self, int index)
        getBody(index) -> Body
        
                Return the bodies that this joint connects. If index is 0 the
                "first" body will be returned, corresponding to the body1
                argument of the attach() method. If index is 1 the "second" body
                will be returned, corresponding to the body2 argument of the
                attach() method.
        
                @param index: Bodx index (0 or 1).
                @type index: int
                
        """
    @staticmethod
    def getFeedback(*args, **kwargs):
        """
        Joint.getFeedback(self)
        getFeedback() -> (force1, torque1, force2, torque2)
        
                Get the forces/torques applied by the joint. If feedback is
                activated (i.e. setFeedback(True) was called) then this method
                returns a tuple (force1, torque1, force2, torque2) with the
                forces and torques applied to body 1 and body 2.  The
                forces/torques are given as 3-tuples.
        
                If feedback is deactivated then the method always returns None.
                
        """
    @staticmethod
    def getKd(*args, **kwargs):
        """
        Joint.getKd(self)
        """
    @staticmethod
    def getKdNumpy(*args, **kwargs):
        """
        Joint.getKdNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getName(*args, **kwargs):
        """
        Joint.getName(self) -> str
        """
    @staticmethod
    def getNumBodies(*args, **kwargs):
        """
        Joint.getNumBodies(self)
        """
    @staticmethod
    def getType(*args, **kwargs):
        """
        Joint.getType(self) -> int
        """
    @staticmethod
    def get_jid(*args, **kwargs):
        """
        Joint.get_jid(self)
        """
    @staticmethod
    def get_joint_dof(*args, **kwargs):
        """
        Joint.get_joint_dof(self) -> int
        """
    @staticmethod
    def isEnabled(*args, **kwargs):
        """
        Joint.isEnabled(self)
        isEnabled() -> bool
        
                Determine whether the joint is enabled. Disabled joints are completely
                ignored during the simulation. Disabled joints don't lose the already
                computed information like anchors and axes.
                
        """
    @staticmethod
    def pre_clear(*args, **kwargs):
        """
        Joint.pre_clear(self)
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        Joint.setAnchor(self, val)
        """
    @staticmethod
    def setAnchorNumpy(*args, **kwargs):
        """
        Joint.setAnchorNumpy(self, ndarray val)
        """
    @staticmethod
    def setFeedback(*args, **kwargs):
        """
        Joint.setFeedback(self, int flag=1)
        setFeedback(flag=True)
        
                Create a feedback buffer. If flag is True then a buffer is
                allocated and the forces/torques applied by the joint can
                be read using the getFeedback() method. If flag is False the
                buffer is released.
        
                @param flag: Specifies whether a buffer should be created or released
                @type flag: bool
                
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        Joint.setKd(self, dReal kdx, dReal kdy, dReal kdz)
        """
    @staticmethod
    def setKd_arrNumpy(*args, **kwargs):
        """
        Joint.setKd_arrNumpy(self, ndarray kd)
        """
    @staticmethod
    def setSameKd(*args, **kwargs):
        """
        Joint.setSameKd(self, dReal kd)
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
class JointGroup:
    """
    JointGroup()
    Joint group.
    
        Constructor::JointGroup()
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        JointGroup.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        JointGroup.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        JointGroup.destroy_immediate(self)
        """
    @staticmethod
    def empty(*args, **kwargs):
        """
        JointGroup.empty(self)
        empty()
        
                Destroy all joints in the group.
                
        """
    def __len__(self):
        """
        Return len(self).
        """
class JointParam:
    ParamBounce: typing.ClassVar[int] = 5
    ParamBounce2: typing.ClassVar[int] = 261
    ParamBounce3: typing.ClassVar[int] = 517
    ParamCFM: typing.ClassVar[int] = 6
    ParamCFM2: typing.ClassVar[int] = 262
    ParamCFM3: typing.ClassVar[int] = 518
    ParamFMax: typing.ClassVar[int] = 3
    ParamFMax2: typing.ClassVar[int] = 259
    ParamFMax3: typing.ClassVar[int] = 515
    ParamFudgeFactor: typing.ClassVar[int] = 4
    ParamFudgeFactor2: typing.ClassVar[int] = 260
    ParamFudgeFactor3: typing.ClassVar[int] = 516
    ParamHiStop: typing.ClassVar[int] = 1
    ParamHiStop2: typing.ClassVar[int] = 257
    ParamHiStop3: typing.ClassVar[int] = 513
    ParamLoStop: typing.ClassVar[int] = 0
    ParamLoStop2: typing.ClassVar[int] = 256
    ParamLoStop3: typing.ClassVar[int] = 512
    ParamStopCFM: typing.ClassVar[int] = 8
    ParamStopCFM2: typing.ClassVar[int] = 264
    ParamStopCFM3: typing.ClassVar[int] = 520
    ParamStopERP: typing.ClassVar[int] = 7
    ParamStopERP2: typing.ClassVar[int] = 263
    ParamStopERP3: typing.ClassVar[int] = 519
    ParamSuspensionCFM: typing.ClassVar[int] = 10
    ParamSuspensionCFM2: typing.ClassVar[int] = 266
    ParamSuspensionCFM3: typing.ClassVar[int] = 522
    ParamSuspensionERP: typing.ClassVar[int] = 9
    ParamSuspensionERP2: typing.ClassVar[int] = 265
    ParamSuspensionERP3: typing.ClassVar[int] = 521
    ParamVel: typing.ClassVar[int] = 2
    ParamVel2: typing.ClassVar[int] = 258
    ParamVel3: typing.ClassVar[int] = 514
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        JointParam.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        JointParam.__setstate_cython__(self, __pyx_state)
        """
class JointTypes:
    Amotor: typing.ClassVar[int] = 9
    Ball: typing.ClassVar[int] = 1
    Contact: typing.ClassVar[int] = 4
    Contact2: typing.ClassVar[int] = 15
    Fixed: typing.ClassVar[int] = 7
    Hinge: typing.ClassVar[int] = 2
    Hinge2: typing.ClassVar[int] = 6
    JointNone: typing.ClassVar[int] = 0
    JointNull: typing.ClassVar[int] = 8
    Lmotor: typing.ClassVar[int] = 10
    PR: typing.ClassVar[int] = 12
    PU: typing.ClassVar[int] = 13
    Piston: typing.ClassVar[int] = 14
    Plane2D: typing.ClassVar[int] = 11
    Slider: typing.ClassVar[int] = 3
    Universal: typing.ClassVar[int] = 5
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        JointTypes.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        JointTypes.__setstate_cython__(self, __pyx_state)
        """
class LMotor(Joint):
    """
    LMotor(World world, JointGroup jointgroup=None)
    LMotor joint.
    
        Constructor::LMotor(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        LMotor.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        LMotor.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getAxis(*args, **kwargs):
        """
        LMotor.getAxis(self, int anum)
        getAxis(anum)
        
                Get an LMotor axis.
        
                @param anum: Axis index (0-2)
                @type anum: int
                
        """
    @staticmethod
    def getAxisNumpy(*args, **kwargs):
        """
        LMotor.getAxisNumpy(self, int anum)
        """
    @staticmethod
    def getNumAxes(*args, **kwargs):
        """
        LMotor.getNumAxes(self)
        getNumAxes() -> int
        
                Get the number of angular axes that are controlled by the LMotor.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        LMotor.getParam(self, int param)
        """
    @staticmethod
    def setAxis(*args, **kwargs):
        """
        LMotor.setAxis(self, int anum, int rel, axis)
        setAxis(anum, rel, axis)
        
                Set an LMotor axis.
        
                The anum argument selects the axis to change (0,1 or 2).
                Each axis can have one of three "relative orientation" modes,
                selected by rel:
        
                0: The axis is anchored to the global frame.
                1: The axis is anchored to the first body.
                2: The axis is anchored to the second body.
        
                @param anum: Axis number
                @param rel: Relative orientation mode
                @param axis: Axis
                @type anum: int
                @type rel: int
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setAxisNumpy(*args, **kwargs):
        """
        LMotor.setAxisNumpy(self, int anum, int rel, ndarray np_axis)
        """
    @staticmethod
    def setNumAxes(*args, **kwargs):
        """
        LMotor.setNumAxes(self, int num)
        setNumAxes(num)
        
                Set the number of angular axes that will be controlled by the LMotor.
                num may be in the range from 0 to 3.
        
                @param num: Number of axes (0-3)
                @type num: int
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        LMotor.setParam(self, int param, dReal value)
        """
class Mass:
    """
    Mass parameters of a rigid body.
    
        This class stores mass parameters of a rigid body which can be
        accessed through the following attributes:
    
        - mass: The total mass of the body (float)
        - c:    The center of gravity position in body frame (3-tuple of floats)
        - I:    The 3x3 inertia tensor in body frame (3-tuple of 3-tuples)
    
        This class wraps the dMass structure from the C API.
    
        @ivar mass: The total mass of the body
        @ivar c: The center of gravity position in body frame (cx, cy, cz)
        @ivar I: The 3x3 inertia tensor in body frame ((I11, I12, I13), (I12, I22, I23), (I13, I23, I33))
        @type mass: float
        @type c: 3-tuple of floats
        @type I: 3-tuple of 3-tuples of floats
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Mass.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Mass.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def add(*args, **kwargs):
        """
        Mass.add(self, Mass b)
        add(b)
        
                Add the mass b to the mass object. Masses can also be added using
                the + operator.
        
                @param b: The mass to add to this mass
                @type b: Mass
                
        """
    @staticmethod
    def adjust(*args, **kwargs):
        """
        Mass.adjust(self, dReal newmass)
        adjust(newmass)
        
                Adjust the total mass. Given mass parameters for some object,
                adjust them so the total mass is now newmass. This is useful
                when using the setXyz() methods to set the mass parameters for
                certain objects - they take the object density, not the total
                mass.
        
                @param newmass: The new total mass
                @type newmass: float
                
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        Mass.copy(self)
        """
    @staticmethod
    def getCNumpy(*args, **kwargs):
        """
        Mass.getCNumpy(self)
        """
    @staticmethod
    def getINumpy(*args, **kwargs):
        """
        Mass.getINumpy(self) -> np.ndarray
        """
    @staticmethod
    def rotate(*args, **kwargs):
        """
        Mass.rotate(self, R)
        """
    @staticmethod
    def rotateNumpy(*args, **kwargs):
        """
        Mass.rotateNumpy(self, ndarray Rot)
        """
    @staticmethod
    def setBox(*args, **kwargs):
        """
        Mass.setBox(self, dReal density, dReal lx, dReal ly, dReal lz)
        setBox(density, lx, ly, lz)
        
                Set the mass parameters to represent a box of the given
                dimensions and density, with the center of mass at (0,0,0)
                relative to the body. The side lengths of the box along the x,
                y and z axes are lx, ly and lz.
        
                @param density: The density of the box
                @param lx: The length along the x axis
                @param ly: The length along the y axis
                @param lz: The length along the z axis
                @type density: float
                @type lx: float
                @type ly: float
                @type lz: float
                
        """
    @staticmethod
    def setBoxTotal(*args, **kwargs):
        """
        Mass.setBoxTotal(self, dReal total_mass, dReal lx, dReal ly, dReal lz)
        setBoxTotal(total_mass, lx, ly, lz)
        
                Set the mass parameters to represent a box of the given
                dimensions and mass, with the center of mass at (0,0,0)
                relative to the body. The side lengths of the box along the x,
                y and z axes are lx, ly and lz.
        
                @param total_mass: The total mass of the box
                @param lx: The length along the x axis
                @param ly: The length along the y axis
                @param lz: The length along the z axis
                @type total_mass: float
                @type lx: float
                @type ly: float
                @type lz: float
                
        """
    @staticmethod
    def setCapsule(*args, **kwargs):
        """
        Mass.setCapsule(self, dReal density, int direction, dReal radius, dReal length)
        setCapsule(density, direction, radius, length)
        
                Set the mass parameters to represent a capsule of the given parameters
                and density, with the center of mass at (0,0,0) relative to the body.
                The radius of the cylinder (and the spherical cap) is radius. The length
                of the cylinder (not counting the spherical cap) is length. The
                cylinder's long axis is oriented along the body's x, y or z axis
                according to the value of direction (1=x, 2=y, 3=z). The first function
                accepts the density of the object, the second accepts its total mass.
        
                @param density: The density of the capsule
                @param direction: The direction of the capsule's cylinder (1=x axis, 2=y axis, 3=z axis)
                @param radius: The radius of the capsule's cylinder
                @param length: The length of the capsule's cylinder (without the caps)
                @type density: float
                @type direction: int
                @type radius: float
                @type length: float
                
        """
    @staticmethod
    def setCapsuleTotal(*args, **kwargs):
        """
        Mass.setCapsuleTotal(self, dReal total_mass, int direction, dReal radius, dReal length)
        setCapsuleTotal(total_mass, direction, radius, length)
        
                Set the mass parameters to represent a capsule of the given parameters
                and mass, with the center of mass at (0,0,0) relative to the body. The
                radius of the cylinder (and the spherical cap) is radius. The length of
                the cylinder (not counting the spherical cap) is length. The cylinder's
                long axis is oriented along the body's x, y or z axis according to the
                value of direction (1=x, 2=y, 3=z). The first function accepts the
                density of the object, the second accepts its total mass.
        
                @param total_mass: The total mass of the capsule
                @param direction: The direction of the capsule's cylinder (1=x axis, 2=y axis, 3=z axis)
                @param radius: The radius of the capsule's cylinder
                @param length: The length of the capsule's cylinder (without the caps)
                @type total_mass: float
                @type direction: int
                @type radius: float
                @type length: float
                
        """
    @staticmethod
    def setCylinder(*args, **kwargs):
        """
        Mass.setCylinder(self, dReal density, int direction, dReal r, dReal h)
        setCylinder(density, direction, r, h)
        
                Set the mass parameters to represent a flat-ended cylinder of
                the given parameters and density, with the center of mass at
                (0,0,0) relative to the body. The radius of the cylinder is r.
                The length of the cylinder is h. The cylinder's long axis is
                oriented along the body's x, y or z axis according to the value
                of direction (1=x, 2=y, 3=z).
        
                @param density: The density of the cylinder
                @param direction: The direction of the cylinder (1=x axis, 2=y axis, 3=z axis)
                @param r: The radius of the cylinder
                @param h: The length of the cylinder
                @type density: float
                @type direction: int
                @type r: float
                @type h: float
                
        """
    @staticmethod
    def setCylinderTotal(*args, **kwargs):
        """
        Mass.setCylinderTotal(self, dReal total_mass, int direction, dReal r, dReal h)
        setCylinderTotal(total_mass, direction, r, h)
        
                Set the mass parameters to represent a flat-ended cylinder of
                the given parameters and mass, with the center of mass at
                (0,0,0) relative to the body. The radius of the cylinder is r.
                The length of the cylinder is h. The cylinder's long axis is
                oriented along the body's x, y or z axis according to the value
                of direction (1=x, 2=y, 3=z).
        
                @param total_mass: The total mass of the cylinder
                @param direction: The direction of the cylinder (1=x axis, 2=y axis, 3=z axis)
                @param r: The radius of the cylinder
                @param h: The length of the cylinder
                @type total_mass: float
                @type direction: int
                @type r: float
                @type h: float
                
        """
    @staticmethod
    def setINumpy(*args, **kwargs):
        """
        Mass.setINumpy(self, ndarray Inertia_in)
        """
    @staticmethod
    def setParameters(*args, **kwargs):
        """
        Mass.setParameters(self, dReal mass, dReal cgx, dReal cgy, dReal cgz, dReal I11, dReal I22, dReal I33, dReal I12, dReal I13, dReal I23)
        setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23)
        
                Set the mass parameters to the given values.
        
                @param mass: Total mass of the body.
                @param cgx: Center of gravity position in the body frame (x component).
                @param cgy: Center of gravity position in the body frame (y component).
                @param cgz: Center of gravity position in the body frame (z component).
                @param I11: Inertia tensor
                @param I22: Inertia tensor
                @param I33: Inertia tensor
                @param I12: Inertia tensor
                @param I13: Inertia tensor
                @param I23: Inertia tensor
                @type mass: float
                @type cgx: float
                @type cgy: float
                @type cgz: float
                @type I11: float
                @type I22: float
                @type I33: float
                @type I12: float
                @type I13: float
                @type I23: float
                
        """
    @staticmethod
    def setSphere(*args, **kwargs):
        """
        Mass.setSphere(self, dReal density, dReal radius)
        setSphere(density, radius)
        
                Set the mass parameters to represent a sphere of the given radius
                and density, with the center of mass at (0,0,0) relative to the body.
        
                @param density: The density of the sphere
                @param radius: The radius of the sphere
                @type density: float
                @type radius: float
                
        """
    @staticmethod
    def setSphereTotal(*args, **kwargs):
        """
        Mass.setSphereTotal(self, dReal total_mass, dReal radius)
        setSphereTotal(total_mass, radius)
        
                Set the mass parameters to represent a sphere of the given radius
                and mass, with the center of mass at (0,0,0) relative to the body.
        
                @param total_mass: The total mass of the sphere
                @param radius: The radius of the sphere
                @type total_mass: float
                @type radius: float
                
        """
    @staticmethod
    def setTriMesh(*args, **kwargs):
        """
        Mass.setTriMesh(self, dReal density, GeomObject g)
        """
    @staticmethod
    def setTriMeshTotal(*args, **kwargs):
        """
        Mass.setTriMeshTotal(self, dReal total_mass, GeomObject g)
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        Mass.setZero(self)
        setZero()
        
                Set all the mass parameters to zero.
        """
    @staticmethod
    def translate(*args, **kwargs):
        """
        Mass.translate(self, t)
        translate(t)
        
                Adjust mass parameters. Given mass parameters for some object,
                adjust them to represent the object displaced by (x,y,z)
                relative to the body frame.
        
                @param t: Translation vector (x, y, z)
                @type t: 3-tuple of floats
                
        """
    def __add__(self, value):
        """
        Return self+value.
        """
    def __radd__(self, value):
        """
        Return value+self.
        """
    def __str__(self):
        """
        Return str(self).
        """
class ODESim:
    class BVHToTarget:
        class BVHToTargetBase:
            """
            
                        Convert bvh motion data to target pose
                        
            """
            @staticmethod
            def calc_facing_quat(target):
                """
                BVHToTargetBase.calc_facing_quat(target)
                """
            def __init__(self, bvh_data: typing.Union[str, VclSimuBackend.pymotionlib.MotionData.MotionData], bvh_fps: int, character, ignore_root_xz_pos: bool = False, bvh_start: typing.Optional[int] = None, bvh_end: typing.Optional[int] = None, set_init_state_as_offset: bool = False, smooth_type: typing.Union[VclSimuBackend.Common.SmoothOperator.GaussianBase, VclSimuBackend.Common.SmoothOperator.ButterWorthBase, NoneType] = None, flip = None):
                """
                BVHToTargetBase.__init__(self, bvh_data: Union[str, pymotionlib.MotionData.MotionData], bvh_fps: int, character, ignore_root_xz_pos: bool = False, bvh_start: Optional[int] = None, bvh_end: Optional[int] = None, set_init_state_as_offset: bool = False, smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase, None] = None, flip=None)
                """
            def body_names(self) -> typing.List[str]:
                """
                BVHToTargetBase.body_names(self) -> List[str]
                
                                get body names
                                
                """
            def calc_body_posi_by_rot(self, quat, joint_posi):
                """
                BVHToTargetBase.calc_body_posi_by_rot(self, quat, joint_posi)
                """
            def calc_posi_by_rot(self, quat, root_posi):
                """
                BVHToTargetBase.calc_posi_by_rot(self, quat, root_posi)
                
                                calculate joints' global position from their global rotation
                                
                """
            def do_smooth(self, smooth_type: typing.Union[VclSimuBackend.Common.SmoothOperator.GaussianBase, VclSimuBackend.Common.SmoothOperator.ButterWorthBase], test_out_fname: typing.Optional[str] = None):
                """
                BVHToTargetBase.do_smooth(self, smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase], test_out_fname: Optional[str] = None)
                """
            def init_all_joint_and_body(self, target):
                """
                BVHToTargetBase.init_all_joint_and_body(self, target)
                
                                joint with root global and local info, all child body info
                                
                """
            def init_end(self, target):
                """
                BVHToTargetBase.init_end(self, target)
                
                                initialize end joints' target info
                                
                """
            def init_facing_coor_end(self, target):
                """
                BVHToTargetBase.init_facing_coor_end(self, target)
                
                                convert bvh end sites to target in facing coordinate
                                
                """
            def init_facing_root(self, target, global_lin_vel: ndarray, global_ang_vel: ndarray):
                """
                BVHToTargetBase.init_facing_root(self, target, ndarray global_lin_vel: np.ndarray, ndarray global_ang_vel: np.ndarray)
                """
            def init_global_child_body(self, target, vel_forward: bool = False):
                """
                BVHToTargetBase.init_global_child_body(self, target, vel_forward: bool = False)
                
                                convert bvh global info to target body in global coordinate
                                
                """
            def init_locally_coor(self, target, vel_forward: bool = False):
                """
                BVHToTargetBase.init_locally_coor(self, target, vel_forward: bool = False)
                
                                convert bvh local rotation to target
                                
                """
            def init_smooth_target(self, target = None, vel_forward: bool = False):
                """
                BVHToTargetBase.init_smooth_target(self, target=None, vel_forward: bool = False)
                """
            def init_target(self, target = None, bvh: typing.Optional[VclSimuBackend.pymotionlib.MotionData.MotionData] = None, vel_forward: bool = False):
                """
                BVHToTargetBase.init_target(self, target=None, bvh: Optional[pymotionlib.MotionData.MotionData] = None, vel_forward: bool = False)
                
                                Note:
                                in ODE engine,
                                a_t = F(x_t, v_t),
                                v_{t + 1} = v_{t} + h * a_{t}
                                x_{t + 1} = x_{t} + h * v_{t + 1}
                                
                """
            def init_target_global(self, target, vel_forward: bool = False):
                """
                BVHToTargetBase.init_target_global(self, target, vel_forward: bool = False)
                
                                initialize target in global coordinate
                                
                """
            def init_target_root(self, target, global_lin_vel: ndarray, global_ang_vel: ndarray):
                """
                BVHToTargetBase.init_target_root(self, target, ndarray global_lin_vel: np.ndarray, ndarray global_ang_vel: np.ndarray)
                
                                Convert bvh to root info
                                This method is OK with root joint
                                
                """
            def joint_names(self) -> typing.List[str]:
                """
                BVHToTargetBase.joint_names(self) -> List[str]
                
                                get joint names
                                
                """
            def only_init_global_target(self, vel_forward: bool = False):
                """
                BVHToTargetBase.only_init_global_target(self, vel_forward: bool = False)
                """
            def set_bvh_offset(self, pos_offset: typing.Optional[numpy.ndarray] = None, quat_offset: typing.Optional[numpy.ndarray] = None):
                """
                BVHToTargetBase.set_bvh_offset(self, pos_offset: Optional[np.ndarray] = None, quat_offset: Optional[np.ndarray] = None)
                """
            def set_init_state_as_bvh_offset(self):
                """
                BVHToTargetBase.set_init_state_as_bvh_offset(self)
                """
            @property
            def bodies(self) -> typing.List[VclSimuBackend.Body]:
                """
                BVHToTargetBase.bodies(self) -> List[Body]
                """
            @property
            def body_info(self):
                """
                BVHToTargetBase.body_info(self)
                
                                get body info
                                
                """
            @property
            def bvh_children(self):
                """
                BVHToTargetBase.bvh_children(self)
                """
            @property
            def bvh_joint_cnt(self):
                """
                BVHToTargetBase.bvh_joint_cnt(self)
                
                                bvh joint count
                                
                """
            @property
            def character_jcnt(self):
                """
                BVHToTargetBase.character_jcnt(self)
                """
            @property
            def character_to_bvh(self):
                """
                BVHToTargetBase.character_to_bvh(self)
                """
            @property
            def child_body_to_joint(self) -> typing.List[int]:
                """
                BVHToTargetBase.child_body_to_joint(self) -> List[int]
                """
            @property
            def end_joint(self):
                """
                BVHToTargetBase.end_joint(self)
                """
            @property
            def end_to_bvh(self):
                """
                BVHToTargetBase.end_to_bvh(self)
                """
            @property
            def frame_cnt(self):
                """
                BVHToTargetBase.frame_cnt(self)
                
                                frame count
                                
                """
            @property
            def has_end_joint(self) -> bool:
                """
                BVHToTargetBase.has_end_joint(self) -> bool
                """
            @property
            def joint_info(self):
                """
                BVHToTargetBase.joint_info(self)
                
                                get joint info
                                
                """
            @property
            def joint_to_child_body(self) -> typing.List[int]:
                """
                BVHToTargetBase.joint_to_child_body(self) -> List[int]
                """
            @property
            def joint_to_parent_body(self) -> typing.List[int]:
                """
                BVHToTargetBase.joint_to_parent_body(self) -> List[int]
                """
            @property
            def joints(self) -> typing.List[typing.Union[VclSimuBackend.Joint, VclSimuBackend.BallJoint, VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint]]:
                """
                BVHToTargetBase.joints(self) -> List[Union[Joint, BallJoint, BallJointAmotor, HingeJoint]]
                """
            @property
            def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
                """
                BVHToTargetBase.root_joint(self) -> Optional[Joint]
                """
            @property
            def space(self) -> SpaceBase:
                """
                BVHToTargetBase.space(self) -> SpaceBase
                """
            @property
            def world(self) -> World:
                """
                BVHToTargetBase.world(self) -> World
                """
    class BodyInfo:
        class BodyInfo:
            __slots__: typing.ClassVar[tuple] = ('world', 'space', 'bodies', 'body_c_id', 'parent', 'children', 'mass_val', 'sum_mass', 'root_body_id', 'initial_inertia', 'visualize_color')
            def __init__(self, world: World, space: SpaceBase):
                """
                BodyInfo.__init__(self, World world: World, SpaceBase space: SpaceBase)
                """
            def __len__(self) -> int:
                """
                BodyInfo.__len__(self) -> int
                
                                length of self.bodies
                                
                """
            def add_body_force(self, force: ndarray):
                """
                BodyInfo.add_body_force(self, ndarray force: np.ndarray)
                """
            def add_body_torque(self, torque: ndarray):
                """
                BodyInfo.add_body_torque(self, ndarray torque: np.ndarray)
                """
            def calc_angular_momentum_slow(self) -> typing.Tuple[numpy.ndarray, VclSimuBackend.Inertia]:
                """
                BodyInfo.calc_angular_momentum_slow(self) -> Tuple[np.ndarray, Inertia]
                
                                angular momentum
                                
                """
            def calc_body_c_id(self) -> ndarray:
                """
                BodyInfo.calc_body_c_id(self) -> np.ndarray
                
                                get pointer for all of bodies. shape == (num_body,).  dtype == np.uint64
                                
                """
            def calc_body_init_inertia(self) -> ndarray:
                """
                BodyInfo.calc_body_init_inertia(self) -> np.ndarray
                
                                Compute the initial inertia for all of bodies
                                
                """
            def calc_body_init_inertia_inv(self) -> ndarray:
                """
                BodyInfo.calc_body_init_inertia_inv(self) -> np.ndarray
                
                                Compute the inverse of initial inertia for all of bodies
                                
                """
            def calc_center_of_mass(self) -> ndarray:
                """
                BodyInfo.calc_center_of_mass(self) -> np.ndarray
                """
            def calc_com_and_facing_com_by_body_state(self, state) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
                """
                BodyInfo.calc_com_and_facing_com_by_body_state(self, state) -> Tuple[np.ndarray, np.ndarray]
                
                                return: com, facing com
                                
                """
            def calc_com_by_body_state(self, state) -> ndarray:
                """
                BodyInfo.calc_com_by_body_state(self, state) -> np.ndarray
                """
            def calc_facing_com_by_body_state(self, state) -> ndarray:
                """
                BodyInfo.calc_facing_com_by_body_state(self, state) -> np.ndarray
                
                                return: np.ndarray in shape (3,)
                                TODO: check with ODE Character
                                
                """
            def calc_facing_quat(self) -> ndarray:
                """
                BodyInfo.calc_facing_quat(self) -> np.ndarray
                
                                return: in shape (4,)
                                
                """
            def calc_geom_ignore_id(self, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True):
                """
                BodyInfo.calc_geom_ignore_id(self, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True)
                
                                Calc ignore id of each geoms in character. ignore collision detection between body and its parent & grandparent
                                :return:
                                
                """
            def calc_momentum(self) -> ndarray:
                """
                BodyInfo.calc_momentum(self) -> np.ndarray
                """
            def calc_sum_mass_pos(self) -> ndarray:
                """
                BodyInfo.calc_sum_mass_pos(self) -> np.ndarray
                """
            def calc_velo_com(self) -> ndarray:
                """
                BodyInfo.calc_velo_com(self) -> np.ndarray
                
                                Calc Velocity of Center of Mass in World Coordinate
                                
                """
            def clear(self):
                """
                BodyInfo.clear(self)
                """
            def get_aabb(self) -> ndarray:
                """
                BodyInfo.get_aabb(self) -> np.ndarray
                
                                Get AABB bounding box of bodies and geoms.
                                
                """
            def get_body_ang_velo(self) -> ndarray:
                """
                BodyInfo.get_body_ang_velo(self) -> np.ndarray
                
                                get all body's angular velocity
                                in shape (num body, 3)
                                
                """
            def get_body_angvel_at(self, index: int) -> ndarray:
                """
                BodyInfo.get_body_angvel_at(self, index: int) -> np.ndarray
                
                                get angular velocity of index-th body. shape == (3,).  dtype == np.float64
                                
                """
            def get_body_contact_mu(self) -> ndarray:
                """
                BodyInfo.get_body_contact_mu(self) -> np.ndarray
                """
            def get_body_force(self) -> ndarray:
                """
                BodyInfo.get_body_force(self) -> np.ndarray
                """
            def get_body_inertia(self) -> ndarray:
                """
                BodyInfo.get_body_inertia(self) -> np.ndarray
                """
            def get_body_pos(self) -> ndarray:
                """
                BodyInfo.get_body_pos(self) -> np.ndarray
                
                                Get all body's position
                                return np.ndarray in shape (num body, 3)
                                
                """
            def get_body_pos_at(self, index: int) -> ndarray:
                """
                BodyInfo.get_body_pos_at(self, index: int) -> np.ndarray
                
                                Get position of index-th body. shape == (3,).  dtype == np.float64
                                
                """
            def get_body_quat(self) -> ndarray:
                """
                BodyInfo.get_body_quat(self) -> np.ndarray
                """
            def get_body_quat_at(self, index: int) -> ndarray:
                """
                BodyInfo.get_body_quat_at(self, index: int) -> np.ndarray
                
                                get quaternion of index-th body. shape == (3,).  dtype == np.float64
                                
                """
            def get_body_rot(self) -> ndarray:
                """
                BodyInfo.get_body_rot(self) -> np.ndarray
                """
            def get_body_rot_mat_at(self, index: int) -> ndarray:
                """
                BodyInfo.get_body_rot_mat_at(self, index: int) -> np.ndarray
                
                                get rotation matrix of index-th body. shape == (9,).  dtype == np.float64
                                
                """
            def get_body_torque(self) -> ndarray:
                """
                BodyInfo.get_body_torque(self) -> np.ndarray
                """
            def get_body_velo(self) -> ndarray:
                """
                BodyInfo.get_body_velo(self) -> np.ndarray
                
                                Get all body's linear velocity
                                return np.ndarray in shape (num body, 3)
                                
                """
            def get_body_velo_at(self, index: int) -> ndarray:
                """
                BodyInfo.get_body_velo_at(self, index: int) -> np.ndarray
                
                                get linear velocity of index-th body. shape == (3,).  dtype == np.float64
                                
                """
            def get_geom_pos(self):
                """
                BodyInfo.get_geom_pos(self)
                """
            def get_geom_rot(self):
                """
                BodyInfo.get_geom_rot(self)
                """
            def get_mirror_index(self) -> typing.List[int]:
                """
                BodyInfo.get_mirror_index(self) -> List[int]
                """
            def get_name_list(self) -> typing.List[str]:
                """
                BodyInfo.get_name_list(self) -> List[str]
                
                                get names for all the bodies
                                
                """
            def get_relative_global_pos(self) -> ndarray:
                """
                BodyInfo.get_relative_global_pos(self) -> np.ndarray
                """
            def get_subset(self, remain_body_index: typing.List[int] = None):
                """
                BodyInfo.get_subset(self, remain_body_index: List[int] = None)
                """
            def init_after_load(self, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True):
                """
                BodyInfo.init_after_load(self, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True)
                """
            def set_body_ang_velo(self, omega: ndarray):
                """
                BodyInfo.set_body_ang_velo(self, ndarray omega: np.ndarray)
                
                                set body angular velocity
                                
                """
            def set_body_pos(self, pos: ndarray):
                """
                BodyInfo.set_body_pos(self, ndarray pos: np.ndarray)
                """
            def set_body_quat(self, quat: ndarray):
                """
                BodyInfo.set_body_quat(self, ndarray quat: np.ndarray)
                """
            def set_body_quat_rot(self, quat: ndarray, rot: ndarray):
                """
                BodyInfo.set_body_quat_rot(self, ndarray quat: np.ndarray, ndarray rot: np.ndarray)
                """
            def set_body_velo(self, velo: ndarray):
                """
                BodyInfo.set_body_velo(self, ndarray velo: np.ndarray)
                """
            @property
            def body0(self) -> typing.Optional[VclSimuBackend.Body]:
                """
                BodyInfo.body0(self) -> Optional[Body]
                
                                Get the 0-th body of the character
                                
                """
            @property
            def body1(self) -> typing.Optional[VclSimuBackend.Body]:
                """
                BodyInfo.body1(self) -> Optional[Body]
                
                                Get the 1-th body of the character
                                
                """
            @property
            def root_body(self) -> Body:
                """
                BodyInfo.root_body(self) -> Body
                """
        @staticmethod
        def my_concatenate(tup: typing.Iterable[numpy.ndarray], axis = 0):
            """
            BodyInfo.my_concatenate(tup: Iterable[np.ndarray], axis=0)
            """
    class BodyInfoState:
        class BodyInfoState:
            """
            
                        save the state for rigid bodies
                        
            """
            __slots__: typing.ClassVar[tuple] = ('pos', 'rot', 'quat', 'linear_vel', 'angular_vel', 'pd_target', 'force', 'torque')
            def __del__(self):
                """
                BodyInfoState.__del__(self)
                """
            def __init__(self):
                """
                BodyInfoState.__init__(self)
                """
            def __len__(self):
                """
                BodyInfoState.__len__(self)
                """
            def calc_delta(self, o):
                """
                BodyInfoState.calc_delta(self, o)
                """
            def cat_to_ndarray(self) -> ndarray:
                """
                BodyInfoState.cat_to_ndarray(self) -> np.ndarray
                """
            def check_delta(self, o):
                """
                BodyInfoState.check_delta(self, o)
                """
            def check_failed(self) -> bool:
                """
                BodyInfoState.check_failed(self) -> bool
                """
            def clear(self):
                """
                BodyInfoState.clear(self)
                """
            def copy(self):
                """
                BodyInfoState.copy(self)
                """
            def is_empty(self):
                """
                BodyInfoState.is_empty(self)
                """
            def load(self, world: World, body_c_id: ndarray):
                """
                BodyInfoState.load(self, World world: World, ndarray body_c_id: np.ndarray)
                """
            def reshape(self):
                """
                BodyInfoState.reshape(self)
                """
            def save(self, world: World, body_c_id: ndarray):
                """
                BodyInfoState.save(self, World world: World, ndarray body_c_id: np.ndarray)
                """
            def set_value(self, pos: ndarray, rot: ndarray, quat: ndarray, linvel: ndarray, angvel: ndarray, pd_target: typing.Optional[numpy.ndarray]):
                """
                BodyInfoState.set_value(self, ndarray pos: np.ndarray, ndarray rot: np.ndarray, ndarray quat: np.ndarray, ndarray linvel: np.ndarray, ndarray angvel: np.ndarray, pd_target: Optional[np.ndarray])
                """
            def to_continuous(self):
                """
                BodyInfoState.to_continuous(self)
                """
    class CharacterJointInfoRoot(ODESim.CharacterWrapper):
        def __init__(self, character):
            """
            CharacterJointInfoRoot.__init__(self, character)
            """
        def get_joint_dof(self) -> ndarray:
            """
            CharacterJointInfoRoot.get_joint_dof(self) -> np.ndarray
            """
        def get_parent_body_index(self) -> ndarray:
            """
            CharacterJointInfoRoot.get_parent_body_index(self) -> np.ndarray
            """
        def get_parent_joint_dof(self) -> ndarray:
            """
            CharacterJointInfoRoot.get_parent_joint_dof(self) -> np.ndarray
            
                        get parent joint dof for each body
                        used in Inverse Dynamics
                        return: np.ndarray in shape (num body,)
                        
            """
        def get_parent_joint_euler_axis(self) -> ndarray:
            """
            CharacterJointInfoRoot.get_parent_joint_euler_axis(self) -> np.ndarray
            
                        return
                        
            """
        def get_parent_joint_euler_order(self) -> typing.List[str]:
            """
            CharacterJointInfoRoot.get_parent_joint_euler_order(self) -> List[str]
            
                        used in Inverse Dynamics
                        return List[str] with length {num body}
                        
            """
        def get_parent_joint_pos(self) -> ndarray:
            """
            CharacterJointInfoRoot.get_parent_joint_pos(self) -> np.ndarray
            
                        Get global position of parent joint of each body
                        used in Inverse Dynamics
                        return: np.ndarray in shape
                        
            """
    class CharacterTOBVH:
        def __init__(self, character, sim_fps: int = 120):
            """
            CharacterTOBVH.__init__(self, character, sim_fps: int = 120)
            """
        def append_no_root_to_buffer(self):
            """
            CharacterTOBVH.append_no_root_to_buffer(self)
            """
        def append_with_root_to_buffer(self):
            """
            CharacterTOBVH.append_with_root_to_buffer(self)
            """
        def build_hierarchy(self):
            """
            CharacterTOBVH.build_hierarchy(self)
            """
        def build_hierarchy_base(self):
            """
            CharacterTOBVH.build_hierarchy_base(self)
            """
        def build_hierarchy_with_root(self):
            """
            CharacterTOBVH.build_hierarchy_with_root(self)
            """
        def bvh_append_no_root(self):
            """
            CharacterTOBVH.bvh_append_no_root(self)
            """
        def bvh_append_with_root(self):
            """
            CharacterTOBVH.bvh_append_with_root(self)
            """
        def bvh_hierarchy_no_root(self):
            """
            CharacterTOBVH.bvh_hierarchy_no_root(self)
            """
        def deepcopy(self):
            """
            CharacterTOBVH.deepcopy(self)
            """
        def forward_kinematics(self, root_pos: ndarray, root_quat: ndarray, joint_local_quat: ndarray) -> pymotionlib.MotionData.MotionData:
            """
            CharacterTOBVH.forward_kinematics(self, ndarray root_pos: np.ndarray, ndarray root_quat: np.ndarray, ndarray joint_local_quat: np.ndarray) -> pymotionlib.MotionData.MotionData
            """
        def insert_end_site(self, motion: typing.Optional[VclSimuBackend.pymotionlib.MotionData.MotionData] = None):
            """
            CharacterTOBVH.insert_end_site(self, motion: Optional[pymotionlib.MotionData.MotionData] = None)
            
                        insert end site to self.motion..
                        
            """
        def merge_buf(self):
            """
            CharacterTOBVH.merge_buf(self)
            """
        def ret_merge_buf(self) -> pymotionlib.MotionData.MotionData:
            """
            CharacterTOBVH.ret_merge_buf(self) -> pymotionlib.MotionData.MotionData
            """
        def to_file(self, fname: unicode = 'test.bvh', print_info = True) -> pymotionlib.MotionData.MotionData:
            """
            CharacterTOBVH.to_file(self, unicode fname: str = u'test.bvh', print_info=True) -> pymotionlib.MotionData.MotionData
            """
        @property
        def root_idx(self):
            """
            CharacterTOBVH.root_idx(self)
            """
    class CharacterWrapper:
        """
        
                Wrapper of ODE Character
                
        """
        def __init__(self, character = None):
            """
            CharacterWrapper.__init__(self, character=None)
            """
        def body_names(self) -> typing.List[str]:
            """
            CharacterWrapper.body_names(self) -> List[str]
            
                        get body names
                        
            """
        def joint_names(self) -> typing.List[str]:
            """
            CharacterWrapper.joint_names(self) -> List[str]
            
                        get joint names
                        
            """
        @property
        def bodies(self) -> typing.List[VclSimuBackend.Body]:
            """
            CharacterWrapper.bodies(self) -> List[Body]
            """
        @property
        def body_info(self):
            """
            CharacterWrapper.body_info(self)
            
                        get body info
                        
            """
        @property
        def child_body_to_joint(self) -> typing.List[int]:
            """
            CharacterWrapper.child_body_to_joint(self) -> List[int]
            """
        @property
        def end_joint(self):
            """
            CharacterWrapper.end_joint(self)
            """
        @property
        def has_end_joint(self) -> bool:
            """
            CharacterWrapper.has_end_joint(self) -> bool
            """
        @property
        def joint_info(self):
            """
            CharacterWrapper.joint_info(self)
            
                        get joint info
                        
            """
        @property
        def joint_to_child_body(self) -> typing.List[int]:
            """
            CharacterWrapper.joint_to_child_body(self) -> List[int]
            """
        @property
        def joint_to_parent_body(self) -> typing.List[int]:
            """
            CharacterWrapper.joint_to_parent_body(self) -> List[int]
            """
        @property
        def joints(self) -> typing.List[typing.Union[VclSimuBackend.Joint, VclSimuBackend.BallJoint, VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint]]:
            """
            CharacterWrapper.joints(self) -> List[Union[Joint, BallJoint, BallJointAmotor, HingeJoint]]
            """
        @property
        def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
            """
            CharacterWrapper.root_joint(self) -> Optional[Joint]
            """
        @property
        def space(self) -> SpaceBase:
            """
            CharacterWrapper.space(self) -> SpaceBase
            """
        @property
        def world(self) -> World:
            """
            CharacterWrapper.world(self) -> World
            """
    class EndJointInfo:
        class EndJointInfo:
            def __init__(self, world: World):
                """
                EndJointInfo.__init__(self, World world: World)
                """
            def __len__(self) -> int:
                """
                EndJointInfo.__len__(self) -> int
                """
            def clear(self):
                """
                EndJointInfo.clear(self)
                """
            def get_global_pos(self) -> ndarray:
                """
                EndJointInfo.get_global_pos(self) -> np.ndarray
                
                                Get End Joint's Global Position
                                
                """
            def resize(self):
                """
                EndJointInfo.resize(self)
                """
    class Environment:
        class Environment:
            """
            
                        static geometry in environment
                        
            """
            def __init__(self, world: World, space: SpaceBase):
                """
                Environment.__init__(self, World world: World, SpaceBase space: SpaceBase)
                """
            def __len__(self) -> int:
                """
                Environment.__len__(self) -> int
                """
            def clear(self):
                """
                Environment.clear(self)
                """
            def create_floor(self, friction = 0.8) -> GeomPlane:
                """
                Environment.create_floor(self, friction=0.8) -> GeomPlane
                """
            def disable(self):
                """
                Environment.disable(self)
                """
            def enable(self):
                """
                Environment.enable(self)
                """
            def get_floor_in_list(self) -> typing.Optional[VclSimuBackend.GeomObject]:
                """
                Environment.get_floor_in_list(self) -> Optional[GeomObject]
                
                                floor will be GeomBox or GeomPlane type..
                                
                """
            def set_space(self, space: typing.Optional[VclSimuBackend.SpaceBase]) -> SpaceBase:
                """
                Environment.set_space(self, space: Optional[SpaceBase]) -> SpaceBase
                """
    class ExtJointList:
        class ExtJointInfo:
            def __init__(self, character0_id: int = 0, body0_id: int = 0, character1_id: int = 0, body1_id: int = 0):
                """
                ExtJointInfo.__init__(self, character0_id: int = 0, body0_id: int = 0, character1_id: int = 0, body1_id: int = 0)
                """
    class JointInfo:
        class JointInfos(ODESim.JointInfo.JointInfosBase):
            @staticmethod
            def body_rotvec(body: Body) -> ndarray:
                """
                JointInfos.body_rotvec(Body body: Body) -> np.ndarray
                
                                Get Body's Rot Vector in world coordinate
                                
                """
            def __init__(self, world: World):
                """
                JointInfos.__init__(self, World world: World)
                """
            def child_qs(self) -> ndarray:
                """
                JointInfos.child_qs(self) -> np.ndarray
                
                                Get Child bodies' quaternion in global coordinate
                                
                """
            def clear(self):
                """
                JointInfos.clear(self)
                """
            def disable_root(self) -> None:
                """
                JointInfos.disable_root(self) -> None
                
                                Disable root joint if exists
                                
                """
            def enable_root(self) -> None:
                """
                JointInfos.enable_root(self) -> None
                
                                enable root joint if exists
                                
                """
            def gen_sample_mask(self, use_joint_names = None) -> ndarray:
                """
                JointInfos.gen_sample_mask(self, use_joint_names=None) -> np.ndarray
                """
            def get_adj_matrix(self) -> ndarray:
                """
                JointInfos.get_adj_matrix(self) -> np.ndarray
                
                                get adj matrix of each joints.
                                
                """
            def get_ball_erp(self, balls: typing.Optional[typing.List[typing.Union[VclSimuBackend.BallJointAmotor, VclSimuBackend.BallJoint]]] = None) -> ndarray:
                """
                JointInfos.get_ball_erp(self, balls: Optional[List[Union[BallJoint, BallJointAmotor]]] = None) -> np.ndarray
                
                                Get erp parameter of all ball joints
                                
                """
            def get_cfm(self) -> ndarray:
                """
                JointInfos.get_cfm(self) -> np.ndarray
                
                                Get CFM parameter of all Joints
                                
                """
            def get_child_body_relative_pos(self) -> ndarray:
                """
                JointInfos.get_child_body_relative_pos(self) -> np.ndarray
                """
            def get_erp(self) -> ndarray:
                """
                JointInfos.get_erp(self) -> np.ndarray
                
                                Get erp of all joints
                                
                """
            def get_global_anchor1(self) -> ndarray:
                """
                JointInfos.get_global_anchor1(self) -> np.ndarray
                
                                call dJointGetBallAnchor1 and dJointGetHingeAnchor1
                                
                """
            def get_global_anchor2(self) -> ndarray:
                """
                JointInfos.get_global_anchor2(self) -> np.ndarray
                
                                call dJointGetBallAnchor2 and dJointGetHingeAnchor2
                                if simulation is totally correct, result of GetAnchor2 should be equal to GetAnchor1
                                
                """
            def get_global_angvels(self) -> ndarray:
                """
                JointInfos.get_global_angvels(self) -> np.ndarray
                
                                return: Joints' angular velocity in global coordinate, in shape (num joint, 3)
                                
                """
            def get_global_pos1(self) -> ndarray:
                """
                JointInfos.get_global_pos1(self) -> np.ndarray
                """
            def get_global_pos2(self) -> ndarray:
                """
                JointInfos.get_global_pos2(self) -> np.ndarray
                
                                get global joint anchor
                                
                """
            def get_hinge_angle(self) -> ndarray:
                """
                JointInfos.get_hinge_angle(self) -> np.ndarray
                
                                return angle of each hinge joint
                                
                """
            def get_hinge_axis1(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_axis1(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_hinge_axis2(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_axis2(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_hinge_erp(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_erp(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                
                                get erp parameter of all hinge joints
                                
                """
            def get_hinge_hi(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_hi(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_hinge_lo(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_lo(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_hinge_raw_axis1(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_raw_axis1(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_hinge_raw_axis2(self, hinges: typing.Optional[typing.List[VclSimuBackend.HingeJoint]] = None) -> ndarray:
                """
                JointInfos.get_hinge_raw_axis2(self, hinges: Optional[List[HingeJoint]] = None) -> np.ndarray
                """
            def get_joint_euler_order(self) -> typing.List[str]:
                """
                JointInfos.get_joint_euler_order(self) -> List[str]
                """
            def get_local_angvels(self, parent_qs_inv: typing.Optional[numpy.ndarray] = None) -> ndarray:
                """
                JointInfos.get_local_angvels(self, parent_qs_inv: Optional[np.ndarray] = None) -> np.ndarray
                
                                param:
                                parent_qs_inv: Optional. inverse of parent bodies' quaternion in global coordinate
                
                                return: Joints' angular velocity in parent body's local coordinate, in shape (num joint, 3)
                                
                """
            def get_local_q(self) -> ndarray:
                """
                JointInfos.get_local_q(self) -> np.ndarray
                
                                joint' quaternion in parent's local coordinate
                                
                """
            def get_mirror_index(self) -> typing.List[int]:
                """
                JointInfos.get_mirror_index(self) -> List[int]
                
                                Modified from Libin Liu's pymotionlib
                                TODO: Test
                                
                """
            def get_neighbours(self) -> typing.List[typing.List[int]]:
                """
                JointInfos.get_neighbours(self) -> List[List[int]]
                """
            def get_parent_body_relative_pos(self) -> ndarray:
                """
                JointInfos.get_parent_body_relative_pos(self) -> np.ndarray
                """
            def get_parent_child_qs(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                """
                JointInfos.get_parent_child_qs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                """
            def get_parent_child_qs_old(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                """
                JointInfos.get_parent_child_qs_old(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                
                                Note: implement of scipy is slow..use cython version instead.
                                return:
                                parent bodies' quaternion in global coordinate,
                                child bodies' quaternion in global coordinate,
                                joint's quaternion in parent's local coordinate.
                                inverse of parent bodies' quaternion in global coordinate
                                
                """
            def get_relative_global_pos(self) -> ndarray:
                """
                JointInfos.get_relative_global_pos(self) -> np.ndarray
                """
            def get_relative_local_pos(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
                """
                JointInfos.get_relative_local_pos(self) -> Tuple[np.ndarray, np.ndarray]
                """
            def get_subset(self, remain_joint_index: typing.List[int]):
                """
                JointInfos.get_subset(self, remain_joint_index: List[int])
                """
            def get_subset_by_name(self, remain_joint_names: typing.List[str]):
                """
                JointInfos.get_subset_by_name(self, remain_joint_names: List[str])
                """
            def parent_qs(self) -> ndarray:
                """
                JointInfos.parent_qs(self) -> np.ndarray
                
                                Get parent bodies' quaternion in global coordinate
                                
                """
            def resize_euler_axis_local(self):
                """
                JointInfos.resize_euler_axis_local(self)
                """
            def set_joint_weights(self, weight_dict: typing.Dict[str, float]):
                """
                JointInfos.set_joint_weights(self, weight_dict: Dict[str, float])
                """
            @property
            def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
                """
                JointInfos.root_joint(self) -> Optional[Joint]
                """
        class JointInfosBase:
            def __add__(self, other):
                """
                JointInfosBase.__add__(self, other)
                """
            def __init__(self, world: World):
                """
                JointInfosBase.__init__(self, World world: World)
                """
            def __len__(self) -> int:
                """
                JointInfosBase.__len__(self) -> int
                """
            def ball_id(self) -> typing.List[int]:
                """
                JointInfosBase.ball_id(self) -> List[int]
                
                                all ball joints' index
                                
                """
            def ball_joints(self) -> typing.List[typing.Union[VclSimuBackend.BallJointAmotor, VclSimuBackend.BallJoint]]:
                """
                JointInfosBase.ball_joints(self) -> List[Union[BallJointAmotor, BallJoint]]
                """
            def clear(self):
                """
                JointInfosBase.clear(self)
                """
            def has_hinge(self) -> bool:
                """
                JointInfosBase.has_hinge(self) -> bool
                """
            def hinge_id(self) -> typing.List[int]:
                """
                JointInfosBase.hinge_id(self) -> List[int]
                
                                All Hinge Joints' index
                                
                """
            def hinge_joints(self) -> typing.List[VclSimuBackend.HingeJoint]:
                """
                JointInfosBase.hinge_joints(self) -> List[HingeJoint]
                
                                All Hinge Joints
                                
                """
            def hinge_lo_hi(self) -> typing.Tuple[typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]:
                """
                JointInfosBase.hinge_lo_hi(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]
                """
            def joint_names(self) -> typing.List[str]:
                """
                JointInfosBase.joint_names(self) -> List[str]
                
                                return: each joints' name
                                
                """
        @staticmethod
        def my_concatenate(tup, axis = 0) -> ndarray:
            """
            JointInfo.my_concatenate(tup, axis=0) -> np.ndarray
            """
    class JointInfoInit:
        class JointInfoInit:
            @staticmethod
            def set_ball_joint_limit(joint: BallJointAmotor, euler_order: unicode, angle_limits: typing.Union[typing.List, numpy.ndarray], raw_axis: typing.Optional[numpy.ndarray] = None):
                """
                JointInfoInit.set_ball_joint_limit(BallJointAmotor joint: BallJointAmotor, unicode euler_order: str, angle_limits: Union[List, np.ndarray], raw_axis: Optional[np.ndarray] = None)
                """
            def __init__(self, joint_info):
                """
                JointInfoInit.__init__(self, joint_info)
                """
            def __len__(self) -> int:
                """
                JointInfoInit.__len__(self) -> int
                """
            def calc_joint_c_id(self):
                """
                JointInfoInit.calc_joint_c_id(self)
                """
            def init_after_load(self):
                """
                JointInfoInit.init_after_load(self)
                """
            def joint_names(self) -> typing.List[str]:
                """
                JointInfoInit.joint_names(self) -> List[str]
                """
            @property
            def euler_axis_local(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoInit.euler_axis_local(self) -> Optional[np.ndarray]
                """
            @euler_axis_local.setter
            def euler_axis_local(self, value: ndarray):
                """
                JointInfoInit.euler_axis_local(self, ndarray value: np.ndarray)
                """
            @property
            def joints(self) -> typing.List[VclSimuBackend.Joint]:
                """
                JointInfoInit.joints(self) -> List[Joint]
                """
            @joints.setter
            def joints(self, value: typing.Optional[VclSimuBackend.Joint]):
                """
                JointInfoInit.joints(self, value: Optional[Joint])
                """
            @property
            def kds(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoInit.kds(self) -> Optional[np.ndarray]
                """
            @property
            def kps(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoInit.kps(self) -> Optional[np.ndarray]
                """
            @property
            def pa_joint_id(self) -> typing.List[int]:
                """
                JointInfoInit.pa_joint_id(self) -> List[int]
                """
            @property
            def world(self) -> World:
                """
                JointInfoInit.world(self) -> World
                """
    class JointInfoWrapper:
        class JointInfoWrapper:
            """
            
                        Wrapper of JointInfos
                        
            """
            def __init__(self, joint_info = None):
                """
                JointInfoWrapper.__init__(self, joint_info=None)
                """
            def __len__(self) -> int:
                """
                JointInfoWrapper.__len__(self) -> int
                """
            def joint_names(self) -> typing.List[str]:
                """
                JointInfoWrapper.joint_names(self) -> List[str]
                """
            @property
            def euler_axis_local(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoWrapper.euler_axis_local(self) -> Optional[np.ndarray]
                """
            @euler_axis_local.setter
            def euler_axis_local(self, value: ndarray):
                """
                JointInfoWrapper.euler_axis_local(self, ndarray value: np.ndarray)
                """
            @property
            def joints(self) -> typing.List[VclSimuBackend.Joint]:
                """
                JointInfoWrapper.joints(self) -> List[Joint]
                """
            @joints.setter
            def joints(self, value: typing.Optional[VclSimuBackend.Joint]):
                """
                JointInfoWrapper.joints(self, value: Optional[Joint])
                """
            @property
            def kds(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoWrapper.kds(self) -> Optional[np.ndarray]
                """
            @property
            def kps(self) -> typing.Optional[numpy.ndarray]:
                """
                JointInfoWrapper.kps(self) -> Optional[np.ndarray]
                """
            @property
            def pa_joint_id(self) -> typing.List[int]:
                """
                JointInfoWrapper.pa_joint_id(self) -> List[int]
                """
            @property
            def world(self) -> World:
                """
                JointInfoWrapper.world(self) -> World
                """
    class JsonCharacterLoader:
        @staticmethod
        def calc_hinge_axis(euler_order: unicode, axis_mat: typing.Optional[numpy.ndarray] = None) -> ndarray:
            """
            JsonCharacterLoader.calc_hinge_axis(unicode euler_order: str, axis_mat: Optional[np.ndarray] = None) -> np.ndarray
            """
        @staticmethod
        def create_joint_base(world: World, json_joint: typing.Dict[str, typing.Any], load_hinge: bool = True, use_ball_limit: bool = True):
            """
            JsonCharacterLoader.create_joint_base(World world: World, json_joint: Dict[str, Any], load_hinge: bool = True, use_ball_limit: bool = True)
            """
        @staticmethod
        def post_create_joint(joint: typing.Union[VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint, VclSimuBackend.BallJoint, VclSimuBackend.FixedJoint], json_joint: typing.Dict[str, typing.Any], load_limits: bool = True):
            """
            JsonCharacterLoader.post_create_joint(joint: Union[BallJointAmotor, HingeJoint, BallJoint, FixedJoint], json_joint: Dict[str, Any], load_limits: bool = True)
            """
        @staticmethod
        def set_ball_limit(joint: BallJointAmotor, euler_order: unicode, angle_limits: typing.Union[typing.List, numpy.ndarray], raw_axis: typing.Optional[numpy.ndarray] = None):
            """
            JsonCharacterLoader.set_ball_limit(BallJointAmotor joint: BallJointAmotor, unicode euler_order: str, angle_limits: Union[List, np.ndarray], raw_axis: Optional[np.ndarray] = None)
            """
        def __init__(self, world: World, space: Space, use_hinge: bool = True, use_angle_limit: bool = True, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True, load_scale: float = 1.0, use_as_base_class: bool = False):
            """
            JsonCharacterLoader.__init__(self, World world: World, space: Space, use_hinge: bool = True, use_angle_limit: bool = True, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True, double load_scale: float = 1.0, use_as_base_class: bool = False)
            
                        Our character model is defined at world coordinate.
                        
            """
        def add_body(self, json_body: typing.Dict[str, typing.Any], update_body_pos_by_com: bool = True) -> Body:
            """
            JsonCharacterLoader.add_body(self, json_body: Dict[str, Any], update_body_pos_by_com: bool = True) -> Body
            
                        @param: recompute_body_pos:
                        return: Body
                        
            """
        def add_joint(self, json_joint: typing.Dict[str, typing.Any], load_hinge: bool = True, load_limits: bool = True) -> Joint:
            """
            JsonCharacterLoader.add_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> Joint
            
                        parse joint info
                        :param json_joint: joint in json format
                        :param load_hinge:
                        :param load_limits:
                        :return: joint
                        
            """
        def body_names(self) -> typing.List[str]:
            """
            JsonCharacterLoader.body_names(self) -> List[str]
            
                        get body names
                        
            """
        def create_geom_object(self, json_geom: typing.Dict[str, typing.Any], calc_mass: bool = True, default_density: float = 1000.0, friction: typing.Optional[float] = None) -> typing.Tuple[VclSimuBackend.GeomObject, typing.Optional[VclSimuBackend.Mass]]:
            """
            JsonCharacterLoader.create_geom_object(self, json_geom: Dict[str, Any], calc_mass: bool = True, double default_density: float = 1000.0, friction: Optional[float] = None) -> Tuple[GeomObject, Optional[Mass]]
            
                        create geometry object
                        
            """
        def create_joint(self, json_joint: typing.Dict[str, typing.Any], load_hinge: bool = True, load_limits: bool = True) -> Joint:
            """
            JsonCharacterLoader.create_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> Joint
            
            
                        :param json_joint:
                        :param load_hinge:
                        :param load_limits:
                        :return:
                        
            """
        def joint_attach(self, joint: Joint, joint_pos, joint_parent: int, joint_child: int):
            """
            JsonCharacterLoader.joint_attach(self, Joint joint: Joint, joint_pos, joint_parent: int, joint_child: int)
            
                        attach bodies to joint
                        
            """
        def joint_names(self) -> typing.List[str]:
            """
            JsonCharacterLoader.joint_names(self) -> List[str]
            
                        get joint names
                        
            """
        def load(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonCharacterLoader.load(self, mess_dict: Dict[str, Any])
            
                        Load ODE Character from json file
                        
            """
        def load_bodies(self, json_bodies: typing.List, update_body_pos_by_com: bool = True):
            """
            JsonCharacterLoader.load_bodies(self, json_bodies: List, update_body_pos_by_com: bool = True)
            
                        Load bodies in json
                        
            """
        def load_endjoints(self, json_endjoints: typing.List):
            """
            JsonCharacterLoader.load_endjoints(self, json_endjoints: List)
            
                        Load end joints in json
                        
            """
        def load_from_file(self, fname: unicode):
            """
            JsonCharacterLoader.load_from_file(self, unicode fname: str)
            
                        Load character from json file
                        
            """
        def load_init_root_info(self, init_root_param: typing.Dict[str, typing.Any]):
            """
            JsonCharacterLoader.load_init_root_info(self, init_root_param: Dict[str, Any])
            """
        def load_joints(self, json_joints: typing.List):
            """
            JsonCharacterLoader.load_joints(self, json_joints: List)
            
                        load joints in json
                        
            """
        def load_pd_control_param(self, json_pd_param: typing.Dict[str, typing.Any]):
            """
            JsonCharacterLoader.load_pd_control_param(self, json_pd_param: Dict[str, Any])
            
                        Load PD Control Param in json
                        
            """
        def parse_ignore_list(self):
            """
            JsonCharacterLoader.parse_ignore_list(self)
            
                        ignore collision detection between some bodies
                        
            """
        def set_character(self, character = None):
            """
            JsonCharacterLoader.set_character(self, character=None)
            """
        @property
        def bodies(self) -> typing.List[VclSimuBackend.Body]:
            """
            JsonCharacterLoader.bodies(self) -> List[Body]
            """
        @property
        def body_info(self):
            """
            JsonCharacterLoader.body_info(self)
            
                        get body info
                        
            """
        @property
        def child_body_to_joint(self) -> typing.List[int]:
            """
            JsonCharacterLoader.child_body_to_joint(self) -> List[int]
            """
        @property
        def end_joint(self):
            """
            JsonCharacterLoader.end_joint(self)
            """
        @property
        def has_end_joint(self) -> bool:
            """
            JsonCharacterLoader.has_end_joint(self) -> bool
            """
        @property
        def joint_info(self):
            """
            JsonCharacterLoader.joint_info(self)
            
                        get joint info
                        
            """
        @property
        def joint_to_child_body(self) -> typing.List[int]:
            """
            JsonCharacterLoader.joint_to_child_body(self) -> List[int]
            """
        @property
        def joint_to_parent_body(self) -> typing.List[int]:
            """
            JsonCharacterLoader.joint_to_parent_body(self) -> List[int]
            """
        @property
        def joints(self) -> typing.List[typing.Union[VclSimuBackend.Joint, VclSimuBackend.BallJoint, VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint]]:
            """
            JsonCharacterLoader.joints(self) -> List[Union[Joint, BallJoint, BallJointAmotor, HingeJoint]]
            """
        @property
        def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
            """
            JsonCharacterLoader.root_joint(self) -> Optional[Joint]
            """
    class JsonSceneLoader:
        """
        
                Load Scene in json format generated from Unity
                
        """
        class AdditionalConfig:
            """
            
                        Additional Configuration in loading json scene
                        
            """
            def __init__(self):
                """
                AdditionalConfig.__init__(self)
                """
            def update_config_dict(self, mess: typing.Optional[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
                """
                AdditionalConfig.update_config_dict(self, mess: Optional[Dict[str, Any]]) -> Dict[str, Any]
                """
        def __init__(self, scene = None, is_running: bool = False):
            """
            JsonSceneLoader.__init__(self, scene=None, is_running: bool = False)
            """
        def file_load(self, fname: unicode, config = None):
            """
            JsonSceneLoader.file_load(self, unicode fname: str, config=None)
            """
        def load_character_list(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonSceneLoader.load_character_list(self, mess_dict: Dict[str, Any])
            """
        def load_environment(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonSceneLoader.load_environment(self, mess_dict: Dict[str, Any])
            """
        def load_ext_forces(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonSceneLoader.load_ext_forces(self, mess_dict: Dict[str, Any])
            
                        Load external forces. such as force from mouse drag/push in Unity Scene
                        
            """
        def load_ext_joints(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonSceneLoader.load_ext_joints(self, mess_dict: Dict[str, Any])
            """
        def load_from_file(self, fname: unicode, config = None):
            """
            JsonSceneLoader.load_from_file(self, unicode fname: str, config=None)
            """
        def load_from_pickle_file(self, fname: unicode, config = None):
            """
            JsonSceneLoader.load_from_pickle_file(self, unicode fname: str, config=None)
            """
        def load_json(self, mess_dict: typing.Dict[str, typing.Any], config: typing.Optional[VclSimuBackend.ODESim.JsonSceneLoader.AdditionalConfig] = None):
            """
            JsonSceneLoader.load_json(self, mess_dict: Dict[str, Any], config: Optional[AdditionalConfig] = None)
            """
        def load_world_attr(self, mess_dict: typing.Dict[str, typing.Any]):
            """
            JsonSceneLoader.load_world_attr(self, mess_dict: Dict[str, Any])
            """
        @property
        def character0(self):
            """
            JsonSceneLoader.character0(self)
            """
        @property
        def characters(self):
            """
            JsonSceneLoader.characters(self)
            """
        @property
        def environment(self):
            """
            JsonSceneLoader.environment(self)
            """
        @property
        def ext_joints(self):
            """
            JsonSceneLoader.ext_joints(self)
            """
        @property
        def space(self) -> SpaceBase:
            """
            JsonSceneLoader.space(self) -> SpaceBase
            """
        @property
        def world(self) -> World:
            """
            JsonSceneLoader.world(self) -> World
            """
    class MeshCharacterLoader:
        def __init__(self, world: World, space: SpaceBase):
            """
            MeshCharacterLoader.__init__(self, World world: World, SpaceBase space: SpaceBase)
            """
        def load_from_obj(self, obj_path, meshname, volume_scale = 1, density_scale = 1, inverse_xaxis = True):
            """
            MeshCharacterLoader.load_from_obj(self, obj_path, meshname, volume_scale=1, density_scale=1, inverse_xaxis=True)
            """
    class MessDictScale:
        @staticmethod
        def handle_dict(mess_dict: typing.Dict[str, typing.Any], load_scale: float):
            """
            MessDictScale.handle_dict(mess_dict: Dict[str, Any], double load_scale: float)
            """
        @staticmethod
        def handle_list(mess_list: typing.List, load_scale: float):
            """
            MessDictScale.handle_list(mess_list: List, double load_scale: float)
            """
        @staticmethod
        def handle_value(key, value, scale: float):
            """
            MessDictScale.handle_value(key, value, double scale: float)
            """
    class ODECharacter:
        class DRootInitInfo:
            def __init__(self) -> None:
                """
                DRootInitInfo.__init__(self) -> None
                """
            def clear(self):
                """
                DRootInitInfo.clear(self)
                """
        class ODECharacter:
            @staticmethod
            def rotate_body_info_state_y_axis(state, angle: float, use_delta_angle: bool = False):
                """
                ODECharacter.rotate_body_info_state_y_axis(state, double angle: float, use_delta_angle: bool = False)
                
                                rotate the BodyInfoState by y axis
                                return: BodyInfoState
                
                                For position, move to the original position, and rotate, then move back
                                For rotation, rotate directly. note that rotation matrix should be recomputed.
                                For linear velocity and angular velocity, rotate directly.
                                Test:
                                After rotate, the simulation result should match
                                
                """
            def __init__(self, world: World, space: SpaceBase):
                """
                ODECharacter.__init__(self, World world: World, SpaceBase space: SpaceBase)
                """
            def calc_kinetic_energy(self) -> ndarray:
                """
                ODECharacter.calc_kinetic_energy(self) -> np.ndarray
                
                                1/2*m*v^2 + 1/2*w^T*I*w
                                
                """
            def cat_root_child_body_value(self, root_value: ndarray, child_body_value: ndarray, dtype = numpy.float64):
                """
                ODECharacter.cat_root_child_body_value(self, ndarray root_value: np.ndarray, ndarray child_body_value: np.ndarray, dtype=np.float64)
                
                                cat value for root body and child body
                                root_value.shape == (batch size, num value)
                                child_body.shape == (batch size, num body - 1, num value)
                                
                """
            def character_facing_coor_com(self, facing_rot_inv: typing.Optional[numpy.ndarray] = None, com: typing.Optional[numpy.ndarray] = None) -> ndarray:
                """
                ODECharacter.character_facing_coor_com(self, facing_rot_inv: Optional[np.ndarray] = None, com: Optional[np.ndarray] = None) -> np.ndarray
                
                                character's CoM in facing coordinate
                                
                """
            def character_facing_coor_com_old(self, facing_rot_inv: typing.Union[scipy.spatial.transform._rotation.Rotation, numpy.ndarray, NoneType] = None) -> ndarray:
                """
                ODECharacter.character_facing_coor_com_old(self, facing_rot_inv: Union[Rotation, np.ndarray, None] = None) -> np.ndarray
                
                                character's CoM in facing coordinate
                                
                """
            def character_facing_coor_com_velo(self) -> ndarray:
                """
                ODECharacter.character_facing_coor_com_velo(self) -> np.ndarray
                
                                character's CoM's velocity in facing coordinate
                                
                """
            def character_facing_coor_end_pos(self, facing_rot_inv: typing.Optional[numpy.ndarray] = None) -> ndarray:
                """
                ODECharacter.character_facing_coor_end_pos(self, facing_rot_inv: Optional[np.ndarray] = None) -> np.ndarray
                """
            def character_facing_coor_end_pos_old(self, facing_rot_inv: typing.Union[scipy.spatial.transform._rotation.Rotation, numpy.ndarray, NoneType] = None) -> ndarray:
                """
                ODECharacter.character_facing_coor_end_pos_old(self, facing_rot_inv: Union[Rotation, np.ndarray, None] = None) -> np.ndarray
                
                                End Joints' Position in character's facing coordinate
                                
                """
            def character_facing_coord_angular_momentum(self) -> ndarray:
                """
                ODECharacter.character_facing_coord_angular_momentum(self) -> np.ndarray
                
                                character's angular momentum in facing coordinate
                                
                """
            def check_root(self):
                """
                ODECharacter.check_root(self)
                
                                check root joint and root body
                                :return:
                                
                """
            def clear(self):
                """
                ODECharacter.clear(self)
                """
            def disable_all_clung_env(self) -> None:
                """
                ODECharacter.disable_all_clung_env(self) -> None
                """
            def enable_all_clung_env(self) -> None:
                """
                ODECharacter.enable_all_clung_env(self) -> None
                """
            def get_aabb(self) -> ndarray:
                """
                ODECharacter.get_aabb(self) -> np.ndarray
                
                                get character aabb
                                
                """
            def get_body_ang_velo(self) -> ndarray:
                """
                ODECharacter.get_body_ang_velo(self) -> np.ndarray
                """
            def get_body_angvel_at(self, index: int) -> ndarray:
                """
                ODECharacter.get_body_angvel_at(self, index: int) -> np.ndarray
                """
            def get_body_mat(self) -> ndarray:
                """
                ODECharacter.get_body_mat(self) -> np.ndarray
                """
            def get_body_name_list(self) -> typing.List[str]:
                """
                ODECharacter.get_body_name_list(self) -> List[str]
                """
            def get_body_pos(self) -> ndarray:
                """
                ODECharacter.get_body_pos(self) -> np.ndarray
                """
            def get_body_pos_at(self, index: int) -> ndarray:
                """
                ODECharacter.get_body_pos_at(self, index: int) -> np.ndarray
                """
            def get_body_quat(self) -> ndarray:
                """
                ODECharacter.get_body_quat(self) -> np.ndarray
                """
            def get_body_quat_at(self, index: int) -> ndarray:
                """
                ODECharacter.get_body_quat_at(self, index: int) -> np.ndarray
                """
            def get_body_rot_mat_at(self, index: int) -> ndarray:
                """
                ODECharacter.get_body_rot_mat_at(self, index: int) -> np.ndarray
                """
            def get_body_velo(self) -> ndarray:
                """
                ODECharacter.get_body_velo(self) -> np.ndarray
                """
            def get_body_velo_at(self, index: int) -> ndarray:
                """
                ODECharacter.get_body_velo_at(self, index: int) -> np.ndarray
                """
            def get_joint_names(self, with_root: bool = False) -> typing.List[str]:
                """
                ODECharacter.get_joint_names(self, with_root: bool = False) -> List[str]
                """
            def get_raw_anchor(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
                """
                ODECharacter.get_raw_anchor(self) -> Tuple[np.ndarray, np.ndarray]
                
                                joint's body1 raw anchor, joint's body2 raw anchor
                                
                """
            def init_root_body_pos(self) -> ndarray:
                """
                ODECharacter.init_root_body_pos(self) -> np.ndarray
                
                                initial root position
                                
                """
            def init_root_quat(self) -> ndarray:
                """
                ODECharacter.init_root_quat(self) -> np.ndarray
                
                                initial root quaternion
                                
                """
            def load(self, body_state):
                """
                ODECharacter.load(self, body_state)
                
                                Load BodyInfoState
                                
                """
            def load_init_state(self) -> None:
                """
                ODECharacter.load_init_state(self) -> None
                
                                load initial state
                                
                """
            def move_character(self, new_pos: ndarray) -> None:
                """
                ODECharacter.move_character(self, ndarray new_pos: np.ndarray) -> None
                """
            def move_character_by_delta(self, delta_pos: ndarray) -> None:
                """
                ODECharacter.move_character_by_delta(self, ndarray delta_pos: np.ndarray) -> None
                """
            def rotate_character(self):
                """
                ODECharacter.rotate_character(self)
                """
            def rotate_y_axis(self, angle: float, use_delta_angle: bool = False):
                """
                ODECharacter.rotate_y_axis(self, double angle: float, use_delta_angle: bool = False)
                """
            def save(self):
                """
                ODECharacter.save(self)
                
                                Save to BodyInfoState
                                
                """
            def save_init_state(self):
                """
                ODECharacter.save_init_state(self)
                
                                Save init state
                                :return: initial state
                                
                """
            def set_body_ang_velo(self, ang_velo: ndarray):
                """
                ODECharacter.set_body_ang_velo(self, ndarray ang_velo: np.ndarray)
                """
            def set_body_pos(self, pos: ndarray):
                """
                ODECharacter.set_body_pos(self, ndarray pos: np.ndarray)
                """
            def set_body_velo(self, velo: ndarray):
                """
                ODECharacter.set_body_velo(self, ndarray velo: np.ndarray)
                """
            def set_character_id(self, new_id: int):
                """
                ODECharacter.set_character_id(self, new_id: int)
                """
            def set_clung_env(self, body_names: typing.Iterable[str], value: bool = True):
                """
                ODECharacter.set_clung_env(self, body_names: Iterable[str], value: bool = True)
                """
            def set_geom_max_friction(self, coef: float = 3.0) -> None:
                """
                ODECharacter.set_geom_max_friction(self, double coef: float = 3.0) -> None
                """
            def set_ode_space(self, space: typing.Optional[VclSimuBackend.SpaceBase]):
                """
                ODECharacter.set_ode_space(self, space: Optional[SpaceBase])
                
                                set space of each geometry in character.
                                
                """
            def set_render_color(self, color: ndarray):
                """
                ODECharacter.set_render_color(self, ndarray color: np.ndarray)
                """
            def set_root_pos(self, pos: ndarray):
                """
                ODECharacter.set_root_pos(self, ndarray pos: np.ndarray)
                """
            @property
            def bodies(self) -> typing.List[VclSimuBackend.Body]:
                """
                ODECharacter.bodies(self) -> List[Body]
                """
            @property
            def end_joint_weights(self) -> typing.Optional[numpy.ndarray]:
                """
                ODECharacter.end_joint_weights(self) -> Optional[np.ndarray]
                """
            @property
            def has_end_joint(self) -> bool:
                """
                ODECharacter.has_end_joint(self) -> bool
                """
            @property
            def is_enable(self) -> bool:
                """
                ODECharacter.is_enable(self) -> bool
                """
            @is_enable.setter
            def is_enable(self, value: bool):
                """
                ODECharacter.is_enable(self, value: bool)
                """
            @property
            def is_kinematic(self) -> bool:
                """
                ODECharacter.is_kinematic(self) -> bool
                """
            @is_kinematic.setter
            def is_kinematic(self, value: bool):
                """
                ODECharacter.is_kinematic(self, value: bool)
                """
            @property
            def joint_weights(self) -> typing.Optional[numpy.ndarray]:
                """
                ODECharacter.joint_weights(self) -> Optional[np.ndarray]
                """
            @property
            def joints(self) -> typing.List[VclSimuBackend.Joint]:
                """
                ODECharacter.joints(self) -> List[Joint]
                """
            @property
            def root_body(self) -> Body:
                """
                ODECharacter.root_body(self) -> Body
                """
            @property
            def root_body_pos(self) -> ndarray:
                """
                ODECharacter.root_body_pos(self) -> np.ndarray
                """
            @property
            def root_body_quat(self) -> ndarray:
                """
                ODECharacter.root_body_quat(self) -> np.ndarray
                """
            @property
            def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
                """
                ODECharacter.root_joint(self) -> Optional[Joint]
                """
            @property
            def self_collision(self) -> bool:
                """
                ODECharacter.self_collision(self) -> bool
                """
            @self_collision.setter
            def self_collision(self, value: bool):
                """
                ODECharacter.self_collision(self, value: bool)
                """
    class ODECharacterInit(ODESim.CharacterWrapper):
        @staticmethod
        def compute_geom_mass_attr(body: Body, create_geom: typing.List[VclSimuBackend.GeomObject], gmasses: typing.List[VclSimuBackend.Mass], gcenters: typing.List, grots: typing.List[scipy.spatial.transform._rotation.Rotation], update_body_pos_by_com: bool = True):
            """
            ODECharacterInit.compute_geom_mass_attr(Body body: Body, create_geom: List[GeomObject], gmasses: List[Mass], gcenters: List, grots: List[Rotation], update_body_pos_by_com: bool = True)
            """
        def __init__(self, character):
            """
            ODECharacterInit.__init__(self, character)
            """
        def add_root_joint(self):
            """
            ODECharacterInit.add_root_joint(self)
            """
        def append_body(self, body: Body, mass_total: Mass, name: unicode, parent: typing.Optional[int]):
            """
            ODECharacterInit.append_body(self, Body body: Body, Mass mass_total: Mass, unicode name: str, parent: Optional[int])
            
                        param:
                        body: Body,
                        mass_total: total mass of body
                        name: body's name
                        idx: body's index
                        parent: body's parent's index
                        
            """
        def calc_height(self):
            """
            ODECharacterInit.calc_height(self)
            
                        compute character's height by AABB bounding box
                        
            """
        def calc_joint_child_body_c_id(self):
            """
            ODECharacterInit.calc_joint_child_body_c_id(self)
            
            
                        
            """
        def calc_joint_parent_body_c_id(self):
            """
            ODECharacterInit.calc_joint_parent_body_c_id(self)
            
            
                        
            """
        def calc_joint_parent_idx(self):
            """
            ODECharacterInit.calc_joint_parent_idx(self)
            
                        Calc parent joint id of each joint.
                        requirement:
                        :return:
                        
            """
        def calc_map_body_joint(self):
            """
            ODECharacterInit.calc_map_body_joint(self)
            
            
                        
            """
        def init_after_load(self, character_id: int = 0, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True):
            """
            ODECharacterInit.init_after_load(self, character_id: int = 0, ignore_parent_collision: bool = True, ignore_grandpa_collision: bool = True)
            
                        initialize character after loading from (xml) configure file
                        
            """
        def init_end_joint(self, names: typing.List[str], parent_body_ids: typing.List[int], end_pos: ndarray):
            """
            ODECharacterInit.init_end_joint(self, names: List[str], parent_body_ids: List[int], ndarray end_pos: np.ndarray)
            
                        initialize end joints
                        
            """
        def init_end_joint_pa_joint_id(self, init_c_id: bool = True):
            """
            ODECharacterInit.init_end_joint_pa_joint_id(self, init_c_id: bool = True)
            
                        Calc parent joint id of each end joint.
                        requirement: self.end_joint.pa_body_id, self.child_body_to_joint
                        :param init_c_id:
                        :return:
                        
            """
        def set_geom_character_id(self, character_id: int = 0):
            """
            ODECharacterInit.set_geom_character_id(self, character_id: int = 0)
            
                        set character_id of each ode GeomObject.
                        used in collision detection: To judge whether one character is collided with other character..
                        
            """
        def set_geom_index(self):
            """
            ODECharacterInit.set_geom_index(self)
            """
        def set_geom_max_friction(self, coef: float = 3.0):
            """
            ODECharacterInit.set_geom_max_friction(self, double coef: float = 3.0)
            """
        def set_has_root(self):
            """
            ODECharacterInit.set_has_root(self)
            """
    class ODEScene:
        class ContactType(enum.IntEnum):
            """
            An enumeration.
            """
            BALL: typing.ClassVar[ODESim.ODEScene.ContactType]  # value = <ContactType.BALL: 2>
            MAX_FORCE_ODE_LCP: typing.ClassVar[ODESim.ODEScene.ContactType]  # value = <ContactType.MAX_FORCE_ODE_LCP: 1>
            ODE_LCP: typing.ClassVar[ODESim.ODEScene.ContactType]  # value = <ContactType.ODE_LCP: 0>
        class ODEScene:
            default_gravity: typing.ClassVar[list] = [0.0, -9.8, 0.0]
            @staticmethod
            def set_falldown_flag(geom1: GeomObject, geom2: GeomObject):
                """
                ODEScene.set_falldown_flag(GeomObject geom1: GeomObject, GeomObject geom2: GeomObject)
                """
            def __init__(self, render_fps: int = 60, sim_fps: int = 120, gravity: typing.Union[typing.Iterable, float, NoneType] = None, friction: float = 0.8, bounce: float = 0.0, self_collision: bool = True, contact_type = None, contact_count: int = 4, extract_contact: bool = True, hand_scene: bool = False):
                """
                ODEScene.__init__(self, render_fps: int = 60, sim_fps: int = 120, gravity: Union[Iterable, float, None] = None, double friction: float = 0.8, double bounce: float = 0.0, self_collision: bool = True, contact_type=None, contact_count: int = 4, extract_contact: bool = True, hand_scene: bool = False)
                """
            def _compute_collide_callback(self, args, geom1: GeomObject, geom2: GeomObject):
                """
                ODEScene._compute_collide_callback(self, args, GeomObject geom1: GeomObject, GeomObject geom2: GeomObject)
                """
            def _generate_contact_joint(self, geom1: GeomObject, geom2: GeomObject, contacts: typing.List[VclSimuBackend.Contact]):
                """
                ODEScene._generate_contact_joint(self, GeomObject geom1: GeomObject, GeomObject geom2: GeomObject, contacts: List[Contact])
                """
            def build_world_and_space(self, gravity: typing.Union[typing.Iterable, float, NoneType] = None):
                """
                ODEScene.build_world_and_space(self, gravity: Union[Iterable, float, None] = None)
                """
            def clear(self, gravity: typing.Union[typing.Iterable, float, NoneType] = None):
                """
                ODEScene.clear(self, gravity: Union[Iterable, float, None] = None)
                
                                clear the scene
                                
                """
            def compute_collide_info(self):
                """
                ODEScene.compute_collide_info(self)
                """
            def contact_basic(self, geom1: GeomObject, geom2: GeomObject) -> typing.Optional[typing.List[VclSimuBackend.Contact]]:
                """
                ODEScene.contact_basic(self, GeomObject geom1: GeomObject, GeomObject geom2: GeomObject) -> Optional[List[Contact]]
                """
            def contact_local_save(self):
                """
                ODEScene.contact_local_save(self)
                
                                we need only save position in body 1 coordinate
                                we can remain global normal vector
                                
                """
            def contact_save(self):
                """
                ODEScene.contact_save(self)
                
                                save contact position, force.
                                render in Unity.
                                
                """
            def create_floor(self) -> GeomPlane:
                """
                ODEScene.create_floor(self) -> GeomPlane
                
                                Create floor geometry
                                
                """
            def damped_simulate(self, n: int = 0):
                """
                ODEScene.damped_simulate(self, n: int = 0)
                """
            def damped_simulate_no_collision(self, n: int = 0):
                """
                ODEScene.damped_simulate_no_collision(self, n: int = 0)
                """
            def damped_simulate_no_collision_once(self):
                """
                ODEScene.damped_simulate_no_collision_once(self)
                """
            def damped_simulate_once(self):
                """
                ODEScene.damped_simulate_once(self)
                """
            def damped_step_fast_collision(self):
                """
                ODEScene.damped_step_fast_collision(self)
                """
            def disable_implicit_damping(self):
                """
                ODEScene.disable_implicit_damping(self)
                """
            def extract_body_contact_label(self) -> ndarray:
                """
                ODEScene.extract_body_contact_label(self) -> np.ndarray
                
                                extract contact label (0/1). here we need not to create contact joints.
                                
                """
            def fast_simulate_once(self):
                """
                ODEScene.fast_simulate_once(self)
                """
            def get_character_id_map(self):
                """
                ODEScene.get_character_id_map(self)
                """
            def near_callback(self, args, geom1: GeomObject, geom2: GeomObject):
                """
                ODEScene.near_callback(self, args, GeomObject geom1: GeomObject, GeomObject geom2: GeomObject)
                """
            def post_simulate_step(self):
                """
                ODEScene.post_simulate_step(self)
                """
            def pre_simulate_step(self) -> JointGroup:
                """
                ODEScene.pre_simulate_step(self) -> JointGroup
                """
            def reset(self):
                """
                ODEScene.reset(self)
                
                                reset each character to initial state
                                
                """
            def resort_geoms(self):
                """
                ODEScene.resort_geoms(self)
                """
            def set_gravity(self, gravity: typing.Union[typing.Iterable, float, NoneType] = None):
                """
                ODEScene.set_gravity(self, gravity: Union[Iterable, float, None] = None)
                """
            def set_render_fps(self, render_fps: int):
                """
                ODEScene.set_render_fps(self, render_fps: int)
                """
            def set_sim_fps(self, sim_fps: int):
                """
                ODEScene.set_sim_fps(self, sim_fps: int)
                """
            def set_simulation_type(self, sim_type):
                """
                ODEScene.set_simulation_type(self, sim_type)
                """
            def simulate(self, n: int = 0):
                """
                ODEScene.simulate(self, n: int = 0)
                """
            def simulate_no_collision(self, n: int = 0):
                """
                ODEScene.simulate_no_collision(self, n: int = 0)
                """
            def simulate_once(self):
                """
                ODEScene.simulate_once(self)
                """
            def step_fast_collision(self):
                """
                ODEScene.step_fast_collision(self)
                """
            def step_range(self) -> range:
                """
                ODEScene.step_range(self) -> range
                """
            def use_implicit_damping(self):
                """
                ODEScene.use_implicit_damping(self)
                """
            @property
            def character0(self):
                """
                ODEScene.character0(self)
                """
            @property
            def contact_count(self) -> int:
                """
                ODEScene.contact_count(self) -> int
                """
            @contact_count.setter
            def contact_count(self, value: int):
                """
                ODEScene.contact_count(self, value: int)
                """
            @property
            def floor(self) -> typing.Optional[VclSimuBackend.GeomPlane]:
                """
                ODEScene.floor(self) -> Optional[GeomPlane]
                """
            @property
            def gravity_numpy(self) -> ndarray:
                """
                ODEScene.gravity_numpy(self) -> np.ndarray
                
                                Get the gravity. default gravity is [0, -9.8, 0]
                                
                """
            @property
            def self_collision(self) -> bool:
                """
                ODEScene.self_collision(self) -> bool
                
                                consider self collision detection is enabled for each character
                                
                """
            @self_collision.setter
            def self_collision(self, value: bool):
                """
                ODEScene.self_collision(self, value: bool)
                """
            @property
            def soft_cfm(self):
                """
                ODEScene.soft_cfm(self)
                
                                cfm value for soft contact
                                
                """
            @soft_cfm.setter
            def soft_cfm(self, value: float):
                """
                ODEScene.soft_cfm(self, double value: float)
                """
            @property
            def soft_erp(self):
                """
                ODEScene.soft_erp(self)
                
                                erp value for soft contact
                                
                """
            @soft_erp.setter
            def soft_erp(self, value: float):
                """
                ODEScene.soft_erp(self, double value: float)
                """
            @property
            def use_soft_contact(self):
                """
                ODEScene.use_soft_contact(self)
                """
            @use_soft_contact.setter
            def use_soft_contact(self, value: bool):
                """
                ODEScene.use_soft_contact(self, value: bool)
                """
        class SceneContactInfo:
            """
            
                        Contact Info Extractor, for visualize in Unity..
                        
            """
            __slots__: typing.ClassVar[tuple] = ('pos', 'force', 'torque', 'geom1_name', 'geom2_name', 'body1_index', 'body2_index', 'contact_label', 'body_contact_force')
            def __init__(self, pos: typing.Union[numpy.ndarray, typing.List, NoneType] = None, force: typing.Union[numpy.ndarray, typing.List, NoneType] = None, geom1_name: typing.Optional[typing.List[str]] = None, geom2_name: typing.Optional[typing.List[str]] = None, contact_label: typing.Union[numpy.ndarray, typing.List[float], NoneType] = None, body_contact_force: typing.Optional[numpy.ndarray] = None):
                """
                SceneContactInfo.__init__(self, pos: Union[np.ndarray, List, None] = None, force: Union[np.ndarray, List, None] = None, geom1_name: Optional[List[str]] = None, geom2_name: Optional[List[str]] = None, contact_label: Union[np.ndarray, List[float], None] = None, body_contact_force: Union[np.ndarray, None] = None)
                """
            def __len__(self) -> int:
                """
                SceneContactInfo.__len__(self) -> int
                """
            def check_delta(self, other):
                """
                SceneContactInfo.check_delta(self, other)
                """
            def clear(self):
                """
                SceneContactInfo.clear(self)
                """
            def merge_force_by_body1(self):
                """
                SceneContactInfo.merge_force_by_body1(self)
                
                                merge the total force by body1 index..
                                There is only one character in the scene
                                
                """
            def out_iter(self):
                """
                SceneContactInfo.out_iter(self)
                
                                get 
                                
                """
            def set_value(self, pos: typing.Optional[numpy.ndarray], force: typing.Optional[numpy.ndarray], geom1_name: typing.Optional[typing.List[str]], geom2_name: typing.Optional[typing.List[str]], contact_label: typing.Union[numpy.ndarray, typing.List[float], NoneType] = None, body_contact_force: typing.Optional[numpy.ndarray] = None):
                """
                SceneContactInfo.set_value(self, pos: Optional[np.ndarray], force: Optional[np.ndarray], geom1_name: Optional[List[str]], geom2_name: Optional[List[str]], contact_label: Union[np.ndarray, List[float], None] = None, body_contact_force: Union[np.ndarray, None] = None)
                """
        class SceneContactLocalInfo:
            """
            
            
                        
            """
            def __init__(self) -> None:
                """
                SceneContactLocalInfo.__init__(self) -> None
                """
            def clear(self):
                """
                SceneContactLocalInfo.clear(self)
                """
            def get_global_pos(self, world: World) -> ndarray:
                """
                SceneContactLocalInfo.get_global_pos(self, World world: World) -> np.ndarray
                """
            def set_value(self, global_pos: ndarray, local_pos: ndarray, normal: ndarray, depth: ndarray, body1_cid: ndarray, to_continuous: bool = True):
                """
                SceneContactLocalInfo.set_value(self, ndarray global_pos: np.ndarray, ndarray local_pos: np.ndarray, ndarray normal: np.ndarray, ndarray depth: np.ndarray, ndarray body1_cid: np.ndarray, to_continuous: bool = True)
                """
        class SimulationType(enum.IntEnum):
            """
            An enumeration.
            """
            DAMPED_FAST_STEP: typing.ClassVar[ODESim.ODEScene.SimulationType]  # value = <SimulationType.DAMPED_FAST_STEP: 1>
            DAMPED_STEP: typing.ClassVar[ODESim.ODEScene.SimulationType]  # value = <SimulationType.DAMPED_STEP: 0>
            FAST_STEP: typing.ClassVar[ODESim.ODEScene.SimulationType]  # value = <SimulationType.FAST_STEP: 3>
            STEP: typing.ClassVar[ODESim.ODEScene.SimulationType]  # value = <SimulationType.STEP: 2>
    class PDController:
        class DampedPDControler:
            """
            
                        using stable PD control.
                        Please refer to [Liu et al. 2013 Simulation and Control of Skeleton-driven Soft Body Characters] for details
                        
            """
            def __init__(self, character):
                """
                DampedPDControler.__init__(self, character)
                """
            def add_torques_by_quat(self, tar_local_qs: ndarray) -> ndarray:
                """
                DampedPDControler.add_torques_by_quat(self, ndarray tar_local_qs: np.ndarray) -> np.ndarray
                """
        class DampedPDControlerSlow(ODESim.PDController.PDControlerBase):
            def __init__(self, joint_info):
                """
                DampedPDControlerSlow.__init__(self, joint_info)
                """
            def add_torques_by_quat(self, tar_local_qs: ndarray) -> ndarray:
                """
                DampedPDControlerSlow.add_torques_by_quat(self, ndarray tar_local_qs: np.ndarray) -> np.ndarray
                
                                Param:
                                tar_local_qs: target joints' quaternion in parent local coordinate
                                
                """
        class PDControler(ODESim.PDController.PDControlerBase):
            def __init__(self, joint_info):
                """
                PDControler.__init__(self, joint_info)
                """
            def add_torques_by_quat(self, tar_local_qs: ndarray) -> ndarray:
                """
                PDControler.add_torques_by_quat(self, ndarray tar_local_qs: np.ndarray) -> np.ndarray
                """
        class PDControlerBase:
            def __init__(self, joint_info):
                """
                PDControlerBase.__init__(self, joint_info)
                """
            def _add_clipped_torque(self, parent_qs: ndarray, local_torques: ndarray) -> ndarray:
                """
                PDControlerBase._add_clipped_torque(self, ndarray parent_qs: np.ndarray, ndarray local_torques: np.ndarray) -> np.ndarray
                
                                Clip torque to avoid Numerical explosion.
                                Param:
                                parent_qs: parent bodies' quaternion in global coordinate
                                local_torques: torques added to joints in parent local coordinate
                                
                """
            def _add_local_torque(self, parent_qs: ndarray, local_torques: ndarray) -> ndarray:
                """
                PDControlerBase._add_local_torque(self, ndarray parent_qs: np.ndarray, ndarray local_torques: np.ndarray) -> np.ndarray
                
                                param: parent_qs: parent bodies' quaternion in global coordinate
                                
                """
            def add_torques_by_quat(self, tar_local_qs: ndarray) -> ndarray:
                """
                PDControlerBase.add_torques_by_quat(self, ndarray tar_local_qs: np.ndarray) -> np.ndarray
                """
    class TargetPose:
        class SetTargetToCharacter:
            """
            
                        use for load {frame} to ODE Character
                        
            """
            def __init__(self, character, target):
                """
                SetTargetToCharacter.__init__(self, character, target)
                """
            def body_names(self) -> typing.List[str]:
                """
                SetTargetToCharacter.body_names(self) -> List[str]
                
                                get body names
                                
                """
            def check(self, frame: int):
                """
                SetTargetToCharacter.check(self, frame: int)
                """
            def joint_names(self) -> typing.List[str]:
                """
                SetTargetToCharacter.joint_names(self) -> List[str]
                
                                get joint names
                                
                """
            def set_character_byframe(self, frame: int = 0, other_character = None):
                """
                SetTargetToCharacter.set_character_byframe(self, frame: int = 0, other_character=None)
                """
            def set_character_byframe_old(self, frame: int = 0, other_character = None):
                """
                SetTargetToCharacter.set_character_byframe_old(self, frame: int = 0, other_character=None)
                
                                load {frame} to ODE Character
                                we don't need to resort joint, because we have joint c id..
                                
                """
            @property
            def bodies(self) -> typing.List[VclSimuBackend.Body]:
                """
                SetTargetToCharacter.bodies(self) -> List[Body]
                """
            @property
            def body_info(self):
                """
                SetTargetToCharacter.body_info(self)
                
                                get body info
                                
                """
            @property
            def child_body_to_joint(self) -> typing.List[int]:
                """
                SetTargetToCharacter.child_body_to_joint(self) -> List[int]
                """
            @property
            def end_joint(self):
                """
                SetTargetToCharacter.end_joint(self)
                """
            @property
            def has_end_joint(self) -> bool:
                """
                SetTargetToCharacter.has_end_joint(self) -> bool
                """
            @property
            def joint_info(self):
                """
                SetTargetToCharacter.joint_info(self)
                
                                get joint info
                                
                """
            @property
            def joint_to_child_body(self) -> typing.List[int]:
                """
                SetTargetToCharacter.joint_to_child_body(self) -> List[int]
                """
            @property
            def joint_to_parent_body(self) -> typing.List[int]:
                """
                SetTargetToCharacter.joint_to_parent_body(self) -> List[int]
                """
            @property
            def joints(self) -> typing.List[typing.Union[VclSimuBackend.Joint, VclSimuBackend.BallJoint, VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint]]:
                """
                SetTargetToCharacter.joints(self) -> List[Union[Joint, BallJoint, BallJointAmotor, HingeJoint]]
                """
            @property
            def num_frames(self):
                """
                SetTargetToCharacter.num_frames(self)
                """
            @property
            def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
                """
                SetTargetToCharacter.root_joint(self) -> Optional[Joint]
                """
            @property
            def space(self) -> SpaceBase:
                """
                SetTargetToCharacter.space(self) -> SpaceBase
                """
            @property
            def world(self) -> World:
                """
                SetTargetToCharacter.world(self) -> World
                """
        class TargetBaseType:
            __slots__: typing.ClassVar[tuple] = ('pos', 'quat', 'rot_mat', 'linvel', 'angvel', 'linacc', 'angacc')
            def __init__(self):
                """
                TargetBaseType.__init__(self)
                """
            def __len__(self) -> int:
                """
                TargetBaseType.__len__(self) -> int
                """
            def __str__(self):
                """
                TargetBaseType.__str__(self)
                """
            def append(self, other):
                """
                TargetBaseType.append(self, other)
                """
            def deepcopy(self):
                """
                TargetBaseType.deepcopy(self)
                """
            def duplicate(self, times: int = 1):
                """
                TargetBaseType.duplicate(self, times: int = 1)
                """
            def resize(self, shape: typing.Union[int, typing.Iterable, typing.Tuple[int]], dtype = numpy.float64):
                """
                TargetBaseType.resize(self, shape: Union[int, Iterable, Tuple[int]], dtype=np.float64)
                """
            def set_value(self, pos: typing.Optional[numpy.ndarray] = None, quat: typing.Optional[numpy.ndarray] = None, rot_mat: typing.Optional[numpy.ndarray] = None, linvel: typing.Optional[numpy.ndarray] = None, angvel: typing.Optional[numpy.ndarray] = None, linacc: typing.Optional[numpy.ndarray] = None, angacc: typing.Optional[numpy.ndarray] = None):
                """
                TargetBaseType.set_value(self, pos: Optional[np.ndarray] = None, quat: Optional[np.ndarray] = None, rot_mat: Optional[np.ndarray] = None, linvel: Optional[np.ndarray] = None, angvel: Optional[np.ndarray] = None, linacc: Optional[np.ndarray] = None, angacc: Optional[np.ndarray] = None)
                """
            def sub_seq(self, start: int = 0, end: typing.Optional[int] = None, skip: typing.Optional[int] = None, is_copy: bool = True):
                """
                TargetBaseType.sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True)
                
                                Get sub sequence of TargetBaseType
                                
                """
            def to_continuous(self):
                """
                TargetBaseType.to_continuous(self)
                """
        class TargetPose:
            __slots__: typing.ClassVar[tuple] = ('globally', 'locally', 'child_body', 'root', 'root_body', 'facing_root', 'end', 'facing_coor_end', 'all_joint_global', 'all_joint_local', 'all_joint_facing', 'all_child_body', 'character_body', 'facing_quat', 'num_frames', 'fps', 'smoothed', 'dup_pos_off_mix', 'dup_rot_off_mix', 'dup_root_pos', 'dup_root_quat')
            def __init__(self):
                """
                TargetPose.__init__(self)
                """
            def __len__(self) -> int:
                """
                TargetPose.__len__(self) -> int
                """
            def append(self, other):
                """
                TargetPose.append(self, other)
                """
            def compute_global_root_dup(self, dup_count: int):
                """
                TargetPose.compute_global_root_dup(self, dup_count: int)
                """
            def compute_global_root_dup_impl(self, dup_count: int, pos_off_mix: typing.Optional[numpy.ndarray], rot_off_mix: typing.Union[numpy.ndarray, scipy.spatial.transform._rotation.Rotation, NoneType]):
                """
                TargetPose.compute_global_root_dup_impl(self, dup_count: int, pos_off_mix: Optional[np.ndarray], rot_off_mix: Union[np.ndarray, Rotation, None])
                """
            def deepcopy(self):
                """
                TargetPose.deepcopy(self)
                """
            def duplicate(self, times: int = 0):
                """
                TargetPose.duplicate(self, times: int = 0)
                """
            def get_facing_body_info(self):
                """
                TargetPose.get_facing_body_info(self)
                """
            def set_dup_off_mix(self, pos_off_mix: ndarray, rot_off_mix: typing.Union[numpy.ndarray, scipy.spatial.transform._rotation.Rotation]):
                """
                TargetPose.set_dup_off_mix(self, ndarray pos_off_mix: np.ndarray, rot_off_mix: Union[np.ndarray, Rotation])
                """
            def sub_seq(self, start: typing.Optional[int] = None, end_: typing.Optional[int] = None, skip: typing.Optional[int] = None, is_copy: bool = True):
                """
                TargetPose.sub_seq(self, start: Optional[int] = None, end_: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True)
                
                                Get sub sequence of TargetPose
                                
                """
            def to_continuous(self):
                """
                TargetPose.to_continuous(self)
                """
    class Utils:
        class BVHJointMap:
            def __init__(self, bvh: pymotionlib.MotionData.MotionData, character):
                """
                BVHJointMap.__init__(self, bvh: pymotionlib.MotionData.MotionData, character)
                """
            def body_names(self) -> typing.List[str]:
                """
                BVHJointMap.body_names(self) -> List[str]
                
                                get body names
                                
                """
            def joint_names(self) -> typing.List[str]:
                """
                BVHJointMap.joint_names(self) -> List[str]
                
                                get joint names
                                
                """
            @property
            def bodies(self) -> typing.List[VclSimuBackend.Body]:
                """
                BVHJointMap.bodies(self) -> List[Body]
                """
            @property
            def body_info(self):
                """
                BVHJointMap.body_info(self)
                
                                get body info
                                
                """
            @property
            def bvh_joint_cnt(self):
                """
                BVHJointMap.bvh_joint_cnt(self)
                
                                bvh joint count
                                
                """
            @property
            def child_body_to_joint(self) -> typing.List[int]:
                """
                BVHJointMap.child_body_to_joint(self) -> List[int]
                """
            @property
            def end_joint(self):
                """
                BVHJointMap.end_joint(self)
                """
            @property
            def has_end_joint(self) -> bool:
                """
                BVHJointMap.has_end_joint(self) -> bool
                """
            @property
            def joint_info(self):
                """
                BVHJointMap.joint_info(self)
                
                                get joint info
                                
                """
            @property
            def joint_to_child_body(self) -> typing.List[int]:
                """
                BVHJointMap.joint_to_child_body(self) -> List[int]
                """
            @property
            def joint_to_parent_body(self) -> typing.List[int]:
                """
                BVHJointMap.joint_to_parent_body(self) -> List[int]
                """
            @property
            def joints(self) -> typing.List[typing.Union[VclSimuBackend.Joint, VclSimuBackend.BallJoint, VclSimuBackend.BallJointAmotor, VclSimuBackend.HingeJoint]]:
                """
                BVHJointMap.joints(self) -> List[Union[Joint, BallJoint, BallJointAmotor, HingeJoint]]
                """
            @property
            def root_joint(self) -> typing.Optional[VclSimuBackend.Joint]:
                """
                BVHJointMap.root_joint(self) -> Optional[Joint]
                """
            @property
            def space(self) -> SpaceBase:
                """
                BVHJointMap.space(self) -> SpaceBase
                """
            @property
            def world(self) -> World:
                """
                BVHJointMap.world(self) -> World
                """
class PRJoint(Joint):
    """
    PRJoint(World world, JointGroup jointgroup=None)
    Prismatic and Rotoide Joint.
    
        Constructor::PRJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        PRJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        PRJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        PRJoint.getAnchor(self)
        getAnchor()
        
                Get a PRJoint anchor.
                
        """
    @staticmethod
    def getAxis1(*args, **kwargs):
        """
        PRJoint.getAxis1(self)
        getAxis1()
        
                Get a PRJoint's prismatic axis.
                
        """
    @staticmethod
    def getAxis2(*args, **kwargs):
        """
        PRJoint.getAxis2(self)
        getAxis2()
        
                Get a PRJoint's rotoide axis.
                
        """
    @staticmethod
    def getPosition(*args, **kwargs):
        """
        PRJoint.getPosition(self)
        getPosition()
        
                Get a PRJoint's linear extension.  (i.e. the prismatic's extension)
                
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        PRJoint.setAnchor(self, pos)
        setAnchor(pos)
        
                Set a PRJoint anchor.
        
                @param pos: Anchor position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis1(*args, **kwargs):
        """
        PRJoint.setAxis1(self, axis)
        setAxis1(axis)
        
                Set a PRJoint's prismatic axis.
        
                @param axis: Axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis2(*args, **kwargs):
        """
        PRJoint.setAxis2(self, axis)
        setAxis2(axis)
        
                Set a PRJoint's rotoide axis.
        
                @param axis: Axis
                @type axis: 3-sequence of floats
                
        """
class Plane2DJoint(Joint):
    """
    Plane2DJoint(World world, JointGroup jointgroup=None)
    Plane-2D Joint.
    
        Constructor::Plane2DJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Plane2DJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Plane2DJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def setAngleParam(*args, **kwargs):
        """
        Plane2DJoint.setAngleParam(self, int param, dReal value)
        """
    @staticmethod
    def setXParam(*args, **kwargs):
        """
        Plane2DJoint.setXParam(self, int param, dReal value)
        """
    @staticmethod
    def setYParam(*args, **kwargs):
        """
        Plane2DJoint.setYParam(self, int param, dReal value)
        """
class PyEigenArrayXXd:
    """
    PyEigenArrayXXd(ndarray a)
    """
    @staticmethod
    def ToNumpy(*args, **kwargs):
        """
        PyEigenArrayXXd.ToNumpy(self) -> np.ndarray
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        PyEigenArrayXXd.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        PyEigenArrayXXd.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        PyEigenArrayXXd.size(self) -> int
        """
class PyEigenArrayXd:
    """
    PyEigenArrayXd(ndarray a)
    """
    @staticmethod
    def ToNumpy(*args, **kwargs):
        """
        PyEigenArrayXd.ToNumpy(self) -> np.ndarray
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        PyEigenArrayXd.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        PyEigenArrayXd.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        PyEigenArrayXd.size(self) -> int
        """
class PyEigenMatrixXd:
    """
    PyEigenMatrixXd(ndarray a)
    """
    @staticmethod
    def ToNumpy(*args, **kwargs):
        """
        PyEigenMatrixXd.ToNumpy(self) -> np.ndarray
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        PyEigenMatrixXd.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        PyEigenMatrixXd.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def cols(*args, **kwargs):
        """
        PyEigenMatrixXd.cols(self) -> int
        """
    @staticmethod
    def rows(*args, **kwargs):
        """
        PyEigenMatrixXd.rows(self) -> int
        """
class PyEigenVectorXd:
    """
    PyEigenVectorXd(ndarray a)
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def ToNumpy(*args, **kwargs):
        """
        PyEigenVectorXd.ToNumpy(self) -> np.ndarray
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        PyEigenVectorXd.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        PyEigenVectorXd.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        PyEigenVectorXd.size(self) -> int
        """
class QuadTreeSpace(SpaceBase):
    """
    QuadTreeSpace(center, extents, depth, SpaceBase space=None)
    Quadtree space.
    
        This uses a pre-allocated hierarchical grid-based AABB tree to
        quickly cull collision checks. It's exceptionally quick for large
        amounts of objects in landscape-shaped worlds. The amount of
        memory used is 4**depth * 32 bytes.
    
        Currently getGeom() is not implemented for the quadtree space.
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        QuadTreeSpace.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        QuadTreeSpace.__setstate_cython__(self, __pyx_state)
        """
class Render:
    class RenderWorld:
        @staticmethod
        def start_record_video():
            """
            RenderWorld.start_record_video()
            """
        def __init__(self, myworld):
            """
            RenderWorld.__init__(self, myworld)
            """
        def check_wid(self):
            """
            RenderWorld.check_wid(self)
            """
        def draw_background(self, x):
            """
            RenderWorld.draw_background(self, x)
            """
        def draw_hingeaxis(self, x):
            """
            RenderWorld.draw_hingeaxis(self, x)
            """
        def draw_localaxis(self, x):
            """
            RenderWorld.draw_localaxis(self, x)
            """
        def get_screen_buffer(self):
            """
            RenderWorld.get_screen_buffer(self)
            
                        We should record video when update function in drawstuff is called..
                        
            """
        def kill(self):
            """
            RenderWorld.kill(self)
            """
        def look_at(self, pos, target, up):
            """
            RenderWorld.look_at(self, pos, target, up)
            """
        def pause(self, time_):
            """
            RenderWorld.pause(self, time_)
            """
        def set_axis_length(self, x):
            """
            RenderWorld.set_axis_length(self, x)
            """
        def set_color(self, col):
            """
            RenderWorld.set_color(self, col)
            """
        def set_joint_radius(self, r):
            """
            RenderWorld.set_joint_radius(self, r)
            """
        def start(self):
            """
            RenderWorld.start(self)
            """
        def track_body(self, body, sync_y):
            """
            RenderWorld.track_body(self, body, sync_y)
            """
class SimpleSpace(SpaceBase):
    """
    SimpleSpace(SpaceBase space=None)
    Simple space.
    
        This does not do any collision culling - it simply checks every
        possible pair of geoms for intersection, and reports the pairs
        whose AABBs overlap. The time required to do intersection testing
        for n objects is O(n**2). This should not be used for large numbers
        of objects, but it can be the preferred algorithm for a small
        number of objects. This is also useful for debugging potential
        problems with the collision system.
        
    """
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        SimpleSpace.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        SimpleSpace.__setstate_cython__(self, __pyx_state)
        """
class SliderJoint(Joint):
    """
    SliderJoint(World world, JointGroup jointgroup=None)
    Slider joint.
        Constructor::SlideJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        SliderJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        SliderJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addForce(*args, **kwargs):
        """
        SliderJoint.addForce(self, force)
        addForce(force)
        
                Applies the given force in the slider's direction.
        
                @param force: Force magnitude
                @type force: float
                
        """
    @staticmethod
    def getAxis(*args, **kwargs):
        """
        SliderJoint.getAxis(self)
        getAxis() -> 3-tuple of floats
        
                Get the slider axis parameter.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        SliderJoint.getParam(self, param)
        """
    @staticmethod
    def getPosition(*args, **kwargs):
        """
        SliderJoint.getPosition(self)
        getPosition() -> float
        
                Get the slider linear position (i.e. the slider's "extension").
        
                When the axis is set, the current position of the attached
                bodies is examined and that position will be the zero
                position.
                
        """
    @staticmethod
    def getPositionRate(*args, **kwargs):
        """
        SliderJoint.getPositionRate(self)
        getPositionRate() -> float
        
                Get the time derivative of the position.
                
        """
    @staticmethod
    def setAxis(*args, **kwargs):
        """
        SliderJoint.setAxis(self, axis)
        setAxis(axis)
        
                Set the slider axis parameter.
        
                @param axis: Slider axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        SliderJoint.setParam(self, param, value)
        """
class SpaceBase(GeomObject):
    """
    SpaceBase(*a, **kw)
    Space class (container for geometry objects).
    
        A Space object is a container for geometry objects which are used
        to do collision detection.
        The space does high level collision culling, which means that it
        can identify which pairs of geometry objects are potentially
        touching.
    
        This Space class can be used for both, a SimpleSpace and a HashSpace
        (see ODE documentation).
    
        >>> space = Space(type=0)   # Create a SimpleSpace
        >>> space = Space(type=1)   # Create a HashSpace
        
    """
    __hash__: typing.ClassVar[None] = None
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def ResortGeoms(*args, **kwargs):
        """
        SpaceBase.ResortGeoms(self)
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        SpaceBase.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        SpaceBase.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def _getData(*args, **kwargs):
        """
        SpaceBase._getData(self)
        """
    @staticmethod
    def _setData(*args, **kwargs):
        """
        SpaceBase._setData(self, value)
        """
    @staticmethod
    def add(*args, **kwargs):
        """
        SpaceBase.add(self, GeomObject geom)
        add(geom)
        
                Add a geom to a space. This does nothing if the geom is
                already in the space.
        
                @param geom: Geom object to add
                @type geom: GeomObject
                
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        SpaceBase.collide(self, arg, callback)
        collide(arg, callback)
        
                Call a callback function one or more times, for all
                potentially intersecting objects in the space. The callback
                function takes 3 arguments:
        
                def NearCallback(arg, geom1, geom2):
        
                The arg parameter is just passed on to the callback function.
                Its meaning is user defined. The geom1 and geom2 arguments are
                the geometry objects that may be near each other. The callback
                function can call the function collide() (not the Space
                method) on geom1 and geom2, perhaps first determining
                whether to collide them at all based on other information.
        
                @param arg: A user argument that is passed to the callback function
                @param callback: Callback function
                @type callback: callable
                
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        SpaceBase.destroy_immediate(self)
        """
    @staticmethod
    def getAllGeomInfos(*args, **kwargs):
        """
        SpaceBase.getAllGeomInfos(self, with_scale_flag=False, with_name_flag=False)
        """
    @staticmethod
    def getAllGeomName(*args, **kwargs):
        """
        SpaceBase.getAllGeomName(self)
        """
    @staticmethod
    def getGeom(*args, **kwargs):
        """
        SpaceBase.getGeom(self, int idx) -> GeomObject
        getGeom(idx) -> GeomObject
        
                Return the geom with the given index contained within the space.
        
                @param idx: Geom index (0,1,...,getNumGeoms()-1)
                @type idx: int
                
        """
    @staticmethod
    def getGeomIDs(*args, **kwargs):
        """
        SpaceBase.getGeomIDs(self) -> np.ndarray
        """
    @staticmethod
    def getPlaceableCount(*args, **kwargs):
        """
        SpaceBase.getPlaceableCount(self)
        """
    @staticmethod
    def getPlaceableGeomInfos(*args, **kwargs):
        """
        SpaceBase.getPlaceableGeomInfos(self)
        """
    @staticmethod
    def get_bodies_aabb(*args, **kwargs):
        """
        SpaceBase.get_bodies_aabb(self, ndarray np_id)
        """
    @staticmethod
    def get_sid(*args, **kwargs):
        """
        SpaceBase.get_sid(self)
        """
    @staticmethod
    def query(*args, **kwargs):
        """
        SpaceBase.query(self, GeomObject geom) -> bool
        query(geom) -> bool
        
                Return True if the given geom is in the space.
        
                @param geom: Geom object to check
                @type geom: GeomObject
                
        """
    @staticmethod
    def remove(*args, **kwargs):
        """
        SpaceBase.remove(self, GeomObject geom)
        remove(geom)
        
                Remove a geom from a space.
        
                @param geom: Geom object to remove
                @type geom: GeomObject
                
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __iter__(self):
        """
        Implement iter(self).
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
class TriMeshData:
    """
    This class stores the mesh data.
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        TriMeshData.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        TriMeshData.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def build(*args, **kwargs):
        """
        TriMeshData.build(self, verts, faces)
        build(verts, faces)
        
                @param verts: Vertices
                @type verts: Sequence of 3-sequences of floats
                @param faces: Face definitions (three indices per face)
                @type faces: Sequence of 3-sequences of ints
                
        """
class UniversalJoint(Joint):
    """
    UniversalJoint(World world, JointGroup jointgroup=None)
    Universal joint.
    
        Constructor::UniversalJoint(world, jointgroup=None)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        UniversalJoint.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        UniversalJoint.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addTorques(*args, **kwargs):
        """
        UniversalJoint.addTorques(self, torque1, torque2)
        addTorques(torque1, torque2)
        
                Applies torque1 about axis 1, and torque2 about axis 2.
        
                @param torque1: Torque 1 magnitude
                @param torque2: Torque 2 magnitude
                @type torque1: float
                @type torque2: float
                
        """
    @staticmethod
    def getAnchor(*args, **kwargs):
        """
        UniversalJoint.getAnchor(self)
        getAnchor() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 1. If the joint is perfectly satisfied, this
                will be the same as the point on body 2.
                
        """
    @staticmethod
    def getAnchor2(*args, **kwargs):
        """
        UniversalJoint.getAnchor2(self)
        getAnchor2() -> 3-tuple of floats
        
                Get the joint anchor point, in world coordinates. This returns
                the point on body 2. If the joint is perfectly satisfied, this
                will be the same as the point on body 1.
                
        """
    @staticmethod
    def getAngle1(*args, **kwargs):
        """
        UniversalJoint.getAngle1(self)
        """
    @staticmethod
    def getAngle1Rate(*args, **kwargs):
        """
        UniversalJoint.getAngle1Rate(self)
        """
    @staticmethod
    def getAngle2(*args, **kwargs):
        """
        UniversalJoint.getAngle2(self)
        """
    @staticmethod
    def getAngle2Rate(*args, **kwargs):
        """
        UniversalJoint.getAngle2Rate(self)
        """
    @staticmethod
    def getAxis1(*args, **kwargs):
        """
        UniversalJoint.getAxis1(self)
        getAxis1() -> 3-tuple of floats
        
                Get the first univeral axis.
                
        """
    @staticmethod
    def getAxis2(*args, **kwargs):
        """
        UniversalJoint.getAxis2(self)
        getAxis2() -> 3-tuple of floats
        
                Get the second univeral axis.
                
        """
    @staticmethod
    def getParam(*args, **kwargs):
        """
        UniversalJoint.getParam(self, int param)
        """
    @staticmethod
    def setAnchor(*args, **kwargs):
        """
        UniversalJoint.setAnchor(self, pos)
        setAnchor(pos)
        
                Set the universal anchor.
        
                @param pos: Anchor position
                @type pos: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis1(*args, **kwargs):
        """
        UniversalJoint.setAxis1(self, axis)
        setAxis1(axis)
        
                Set the first universal axis. Axis 1 and axis 2 should be
                perpendicular to each other.
        
                @param axis: Joint axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setAxis2(*args, **kwargs):
        """
        UniversalJoint.setAxis2(self, axis)
        setAxis2(axis)
        
                Set the second universal axis. Axis 1 and axis 2 should be
                perpendicular to each other.
        
                @param axis: Joint axis
                @type axis: 3-sequence of floats
                
        """
    @staticmethod
    def setParam(*args, **kwargs):
        """
        UniversalJoint.setParam(self, int param, dReal value)
        """
class Utils:
    class InvDynAttr:
        def __init__(self, ref_start: int, ref_end: int) -> None:
            """
            InvDynAttr.__init__(self, ref_start: int, ref_end: int) -> None
            """
    class MergeMode(enum.IntEnum):
        """
        An enumeration.
        """
        all_body: typing.ClassVar[Utils.MergeMode]  # value = <MergeMode.all_body: 1>
        only_root: typing.ClassVar[Utils.MergeMode]  # value = <MergeMode.only_root: 0>
    class SliceChangeMode(enum.IntEnum):
        """
        An enumeration.
        """
        behind: typing.ClassVar[Utils.SliceChangeMode]  # value = <SliceChangeMode.behind: 1>
        front: typing.ClassVar[Utils.SliceChangeMode]  # value = <SliceChangeMode.front: 0>
    @staticmethod
    def smooth_motion_data(bvh: pymotionlib.MotionData.MotionData, smooth_type: typing.Union[VclSimuBackend.Common.SmoothOperator.GaussianBase, VclSimuBackend.Common.SmoothOperator.ButterWorthBase], test_out_fname: typing.Optional[str] = None, smooth_position: bool = True, smooth_rotation: bool = True):
        """
        Utils.smooth_motion_data(bvh: pymotionlib.MotionData.MotionData, smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase], test_out_fname: Optional[str] = None, smooth_position: bool = True, smooth_rotation: bool = True)
        """
class World:
    """
    Dynamics world.
        The world object is a container for rigid bodies and joints.
        Constructor::World()
        
    """
    __hash__: typing.ClassVar[None] = None
    __pyx_vtable__: typing.ClassVar[typing.Any]  # value = <capsule object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        World.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        World.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def addBodyForce(*args, **kwargs):
        """
        World.addBodyForce(self, ndarray np_id, ndarray np_force)
        """
    @staticmethod
    def addBodyTorque(*args, **kwargs):
        """
        World.addBodyTorque(self, ndarray np_id, ndarray np_tor)
        """
    @staticmethod
    def add_global_torque(*args, **kwargs):
        """
        World.add_global_torque(self, ndarray global_torque, ndarray parent_body_id, ndarray child_body_id)
        """
    @staticmethod
    def bodyListToNumpy(*args, **kwargs):
        """
        World.bodyListToNumpy(self, list body_list)
        """
    @staticmethod
    def compute_body_com(*args, **kwargs):
        """
        World.compute_body_com(self, ndarray np_id)
        """
    @staticmethod
    def createBallJoint(*args, **kwargs):
        """
        World.createBallJoint(self, jointgroup=None)
        """
    @staticmethod
    def createBody(*args, **kwargs):
        """
        World.createBody(self)
        """
    @staticmethod
    def createContactJoint(*args, **kwargs):
        """
        World.createContactJoint(self, jointgroup, contact)
        """
    @staticmethod
    def createContactJointMaxForce(*args, **kwargs):
        """
        World.createContactJointMaxForce(self, jointgroup, contact)
        """
    @staticmethod
    def createFixedJoint(*args, **kwargs):
        """
        World.createFixedJoint(self, jointgroup=None)
        """
    @staticmethod
    def createHinge2Joint(*args, **kwargs):
        """
        World.createHinge2Joint(self, jointgroup=None)
        """
    @staticmethod
    def createHingeJoint(*args, **kwargs):
        """
        World.createHingeJoint(self, jointgroup=None)
        """
    @staticmethod
    def createSliderJoint(*args, **kwargs):
        """
        World.createSliderJoint(self, jointgroup=None)
        """
    @staticmethod
    def dampedStep(*args, **kwargs):
        """
        World.dampedStep(self, dReal stepsize)
        """
    @staticmethod
    def damped_step_fast_collision(*args, **kwargs):
        """
        World.damped_step_fast_collision(self, SpaceBase space, dReal stepsize)
        """
    @staticmethod
    def destroy_immediate(*args, **kwargs):
        """
        World.destroy_immediate(self)
        """
    @staticmethod
    def getAllBodyID(*args, **kwargs):
        """
        World.getAllBodyID(self) -> np.ndarray
        """
    @staticmethod
    def getAllBodyInfos(*args, **kwargs):
        """
        World.getAllBodyInfos(self)
        """
    @staticmethod
    def getAngularDamping(*args, **kwargs):
        """
        World.getAngularDamping(self) -> dReal
        getAngularDamping() -> float
        
                Get the world's angular damping scale.
                
        """
    @staticmethod
    def getBallAndHingeAnchor1(*args, **kwargs):
        """
        World.getBallAndHingeAnchor1(self, ndarray np_id)
        """
    @staticmethod
    def getBallAndHingeAnchor2(*args, **kwargs):
        """
        World.getBallAndHingeAnchor2(self, ndarray np_id)
        """
    @staticmethod
    def getBallAndHingeAnchorAvg(*args, **kwargs):
        """
        World.getBallAndHingeAnchorAvg(self, ndarray np_id)
        """
    @staticmethod
    def getBallAndHingeInfos(*args, **kwargs):
        """
        World.getBallAndHingeInfos(self)
        """
    @staticmethod
    def getBallAndHingeRawAnchor(*args, **kwargs):
        """
        World.getBallAndHingeRawAnchor(self, ndarray np_id)
        """
    @staticmethod
    def getBallAndHingeRawAnchor1(*args, **kwargs):
        """
        World.getBallAndHingeRawAnchor1(self, ndarray np_id)
        """
    @staticmethod
    def getBallAndHingeRawAnchor2(*args, **kwargs):
        """
        World.getBallAndHingeRawAnchor2(self, ndarray np_id)
        """
    @staticmethod
    def getBodyAngVel(*args, **kwargs):
        """
        World.getBodyAngVel(self, ndarray np_id)
        """
    @staticmethod
    def getBodyForce(*args, **kwargs):
        """
        World.getBodyForce(self, ndarray np_id)
        """
    @staticmethod
    def getBodyGeomCount(*args, **kwargs):
        """
        World.getBodyGeomCount(self, ndarray np_id) -> int
        """
    @staticmethod
    def getBodyGeomIDs(*args, **kwargs):
        """
        World.getBodyGeomIDs(self, ndarray np_id) -> np.ndarray
        """
    @staticmethod
    def getBodyInertia(*args, **kwargs):
        """
        World.getBodyInertia(self, ndarray np_id)
        """
    @staticmethod
    def getBodyInertiaInv(*args, **kwargs):
        """
        World.getBodyInertiaInv(self, ndarray np_id)
        """
    @staticmethod
    def getBodyInfos(*args, **kwargs):
        """
        World.getBodyInfos(self, ndarray np_id)
        """
    @staticmethod
    def getBodyLinVel(*args, **kwargs):
        """
        World.getBodyLinVel(self, ndarray np_id)
        """
    @staticmethod
    def getBodyPos(*args, **kwargs):
        """
        World.getBodyPos(self, ndarray np_id)
        """
    @staticmethod
    def getBodyQuatScipy(*args, **kwargs):
        """
        World.getBodyQuatScipy(self, ndarray np_id)
        """
    @staticmethod
    def getBodyRot(*args, **kwargs):
        """
        World.getBodyRot(self, ndarray np_id)
        """
    @staticmethod
    def getBodyTorque(*args, **kwargs):
        """
        World.getBodyTorque(self, ndarray np_id)
        """
    @staticmethod
    def getGravity(*args, **kwargs):
        """
        World.getGravity(self)
        getGravity() -> 3-tuple
        
                Return the world's global gravity vector as a 3-tuple of floats.
                
        """
    @staticmethod
    def getGravityNumpy(*args, **kwargs):
        """
        World.getGravityNumpy(self) -> np.ndarray
        """
    @staticmethod
    def getLinearDamping(*args, **kwargs):
        """
        World.getLinearDamping(self) -> dReal
        getLinearDamping() -> float
        
                Get the world's linear damping scale.
                
        """
    @staticmethod
    def getNumBallAndHingeJoints(*args, **kwargs):
        """
        World.getNumBallAndHingeJoints(self)
        """
    @staticmethod
    def get_all_hinge_angle(*args, **kwargs):
        """
        World.get_all_hinge_angle(self, ndarray hinge_id)
        """
    @staticmethod
    def get_all_joint_local_angle(*args, **kwargs):
        """
        World.get_all_joint_local_angle(self, ndarray joint_id)
        """
    @staticmethod
    def get_pd_control_torque(*args, **kwargs):
        """
        World.get_pd_control_torque(self, ndarray joint_id, ndarray local_target_quat_in: np.ndarray, ndarray kps_in: np.ndarray, ndarray tor_lim_in: np.ndarray)
        """
    @staticmethod
    def get_wid(*args, **kwargs):
        """
        World.get_wid(self)
        """
    @staticmethod
    def impulseToForce(*args, **kwargs):
        """
        World.impulseToForce(self, dReal stepsize, impulse)
        impulseToForce(stepsize, impulse) -> 3-tuple
        
                If you want to apply a linear or angular impulse to a rigid
                body, instead of a force or a torque, then you can use this
                function to convert the desired impulse into a force/torque
                vector before calling the dBodyAdd... function.
        
                @param stepsize: Time step
                @param impulse: Impulse vector
                @type stepsize: float
                @type impulse: 3-tuple of floats
                
        """
    @staticmethod
    def jointListToNumpy(*args, **kwargs):
        """
        World.jointListToNumpy(self, list joint_list)
        """
    @staticmethod
    def loadBodyAngVel(*args, **kwargs):
        """
        World.loadBodyAngVel(self, ndarray np_id, ndarray np_ang_vel)
        """
    @staticmethod
    def loadBodyForce(*args, **kwargs):
        """
        World.loadBodyForce(self, ndarray np_id, ndarray np_force)
        """
    @staticmethod
    def loadBodyInfos(*args, **kwargs):
        """
        World.loadBodyInfos(self, ndarray np_id, ndarray np_pos, ndarray np_quat, ndarray np_rot, ndarray np_lin_vel, ndarray np_ang_vel, ndarray np_force, ndarray np_torque)
        """
    @staticmethod
    def loadBodyLinVel(*args, **kwargs):
        """
        World.loadBodyLinVel(self, ndarray np_id, ndarray np_lin_vel)
        """
    @staticmethod
    def loadBodyPos(*args, **kwargs):
        """
        World.loadBodyPos(self, ndarray np_id, ndarray np_pos)
        """
    @staticmethod
    def loadBodyQuat(*args, **kwargs):
        """
        World.loadBodyQuat(self, ndarray np_id, ndarray np_quat)
        """
    @staticmethod
    def loadBodyQuatAndRotNoNorm(*args, **kwargs):
        """
        World.loadBodyQuatAndRotNoNorm(self, ndarray np_id, ndarray np_quat, ndarray np_rot)
        """
    @staticmethod
    def loadBodyTorque(*args, **kwargs):
        """
        World.loadBodyTorque(self, ndarray np_id, ndarray np_torque)
        """
    @staticmethod
    def quickStep(*args, **kwargs):
        """
        World.quickStep(self, dReal stepsize)
        quickStep(stepsize)
        
                Step the world. This uses an iterative method that takes time
                on the order of O(m*N) and memory on the order of O(m), where m is
                the total number of constraint rows and N is the number of
                iterations.
        
                For large systems this is a lot faster than dWorldStep, but it
                is less accurate.
        
                @param stepsize: Time step
                @type stepsize: float
                
        """
    @staticmethod
    def setAngularDamping(*args, **kwargs):
        """
        World.setAngularDamping(self, dReal scale)
        setAngularDamping(scale)
        
                Set the world's angular damping scale.
                        @param scale The angular damping scale that is to be applied to bodies.
                        Default is 0 (no damping). Should be in the interval [0, 1].
                @type scale: float
                
        """
    @staticmethod
    def setGravity(*args, **kwargs):
        """
        World.setGravity(self, gravity)
        setGravity(gravity)
        
                Set the world's global gravity vector.
        
                @param gravity: Gravity vector
                @type gravity: 3-sequence of floats
                
        """
    @staticmethod
    def setGravityYEarth(*args, **kwargs):
        """
        World.setGravityYEarth(self)
        """
    @staticmethod
    def setLinearDamping(*args, **kwargs):
        """
        World.setLinearDamping(self, dReal scale)
        setLinearDamping(scale)
        
                Set the world's linear damping scale.
                        @param scale The linear damping scale that is to be applied to bodies.
                        Default is 0 (no damping). Should be in the interval [0, 1].
                @type scale: float
                
        """
    @staticmethod
    def step(*args, **kwargs):
        """
        World.step(self, dReal stepsize)
        step(stepsize)
        
                Step the world. This uses a "big matrix" method that takes
                time on the order of O(m3) and memory on the order of O(m2), where m
                is the total number of constraint rows.
        
                For large systems this will use a lot of memory and can be
                very slow, but this is currently the most accurate method.
        
                @param stepsize: Time step
                @type stepsize: float
                
        """
    @staticmethod
    def step_fast_collision(*args, **kwargs):
        """
        World.step_fast_collision(self, SpaceBase space, dReal stepsize)
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
class _GeomAttrs:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        _GeomAttrs.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        _GeomAttrs.__setstate_cython__(self, __pyx_state)
        """
class _SpaceIterator2:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __next__(*args, **kwargs):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        _SpaceIterator2.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        _SpaceIterator2.__setstate_cython__(self, __pyx_state)
        """
    def __iter__(self):
        """
        Implement iter(self).
        """
class pymotionlib:
    class BVHLoader:
        @staticmethod
        def load(fn: unicode, insert_T_pose: bool = False, ignore_root_offset = True, max_frames = None, ignore_root_xz_pos = False):
            """
            BVHLoader.load(unicode fn: str, insert_T_pose: bool = False, ignore_root_offset=True, max_frames=None, ignore_root_xz_pos=False)
            """
        @staticmethod
        def load_from_io(f, fn = '', insert_T_pose: bool = False, ignore_root_offset = True, max_frames = None, ignore_root_xz_pos = False):
            """
            BVHLoader.load_from_io(f, fn=u'', insert_T_pose: bool = False, ignore_root_offset=True, max_frames=None, ignore_root_xz_pos=False)
            """
        @staticmethod
        def load_from_string(bvh_str: unicode, insert_T_pose: bool = False, ignore_root_offset = True, max_frames = None, ignore_root_xz_pos = False):
            """
            BVHLoader.load_from_string(unicode bvh_str: str, insert_T_pose: bool = False, ignore_root_offset=True, max_frames=None, ignore_root_xz_pos=False)
            """
        @staticmethod
        def save(data, fn: unicode, fmt: unicode = '%10.6f', euler_order: unicode = 'XYZ', translational_joints = False, insert_T_pose: bool = False):
            """
            BVHLoader.save(data, unicode fn: str, unicode fmt: str = u'%10.6f', unicode euler_order: str = u'XYZ', translational_joints=False, insert_T_pose: bool = False)
            """
        @staticmethod
        def save_as_string(data, fmt: unicode = '%10.6f', euler_order: unicode = 'XYZ', translational_joints = False, insert_T_pose: bool = False):
            """
            BVHLoader.save_as_string(data, unicode fmt: str = u'%10.6f', unicode euler_order: str = u'XYZ', translational_joints=False, insert_T_pose: bool = False)
            """
        @staticmethod
        def save_to_io(data, f, fmt: unicode = '%10.6f', euler_order: unicode = 'XYZ', translational_joints = False, insert_T_pose: bool = False):
            """
            BVHLoader.save_to_io(data, f, unicode fmt: str = u'%10.6f', unicode euler_order: str = u'XYZ', translational_joints=False, insert_T_pose: bool = False)
            """
    class ExtEndSite:
        @staticmethod
        def load_no_end_site(f: typing.IO, ignore_root_offset = True, max_frames = None):
            """
            ExtEndSite.load_no_end_site(f: IO, ignore_root_offset=True, max_frames=None)
            """
        @staticmethod
        def save_ext_end_site(data, f: typing.IO, fmt: unicode = '%10.6f', euler_order: unicode = 'XYZ', ext_end_site: typing.Optional[typing.Dict[int, numpy.ndarray]] = None):
            """
            ExtEndSite.save_ext_end_site(data, f: IO, unicode fmt: str = u'%10.6f', unicode euler_order: str = u'XYZ', ext_end_site: Optional[Dict[int, np.ndarray]] = None)
            """
    class MotionData:
        class MotionData:
            __slots__: typing.ClassVar[tuple] = ('_skeleton_joints', '_skeleton_joint_parents', '_skeleton_joint_offsets', '_end_sites', '_num_joints', '_num_frames', '_fps', '_joint_rotation', '_joint_translation', '_joint_position', '_joint_orientation')
            def __init__(self) -> None:
                """
                MotionData.__init__(self) -> None
                """
            def align_joint_rotation_representation(self):
                """
                MotionData.align_joint_rotation_representation(self)
                 make sure that the quaternions are aligned
                                
                """
            def append(self, other_):
                """
                MotionData.append(self, other_)
                """
            def append_trans_rotation(self, trans: ndarray, rotation: ndarray):
                """
                MotionData.append_trans_rotation(self, ndarray trans: np.ndarray, ndarray rotation: np.ndarray)
                """
            def compute_angular_velocity(self, forward: bool = False):
                """
                MotionData.compute_angular_velocity(self, forward: bool = False)
                 compute angular velocities of every joint using finite difference
                
                                    the velocities are in the world coordinates
                
                                    forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                                        otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n-1.inv()
                
                                    return: an array of size (num_frame, num_joint, 3),
                                        for forward/backward difference, the last/first frame is the
                                        frame next to it
                                
                """
            def compute_joint_global_info(self, joint_translation: ndarray, joint_rotation: ndarray, joint_position: ndarray = None, joint_orientation: ndarray = None):
                """
                MotionData.compute_joint_global_info(self, ndarray joint_translation: np.ndarray, ndarray joint_rotation: np.ndarray, ndarray joint_position: np.ndarray = None, ndarray joint_orientation: np.ndarray = None)
                 compute global information based on given local information
                                
                """
            def compute_joint_local_info(self, joint_position: ndarray, joint_orientation: ndarray, joint_translation: ndarray = None, joint_rotation: ndarray = None):
                """
                MotionData.compute_joint_local_info(self, ndarray joint_position: np.ndarray, ndarray joint_orientation: np.ndarray, ndarray joint_translation: np.ndarray = None, ndarray joint_rotation: np.ndarray = None)
                 compute local information based on given global information
                                
                """
            def compute_linear_velocity(self, forward: bool = False):
                """
                MotionData.compute_linear_velocity(self, forward: bool = False)
                 compute linear velocities of every joint using finite difference
                
                                    the velocities are in the world coordinates
                
                                    return: an array of size (num_frame, num_joint, 3),
                                        for forward/backward difference, the last/first frame is the
                                        frame next to it
                                
                """
            def compute_rotational_speed(self, forward: bool):
                """
                MotionData.compute_rotational_speed(self, forward: bool)
                 compute the `local` rotational speed of every joint using finite difference
                
                                    note that different from `compute_angular_velocity`, this is the relative
                                    speed of joints wrt. their parents, and the values are represented in the
                                    parents' local coordinates
                
                                    forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                                        otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n.inv()
                
                                    return: an array of size (num_frame, num_joint, 3),
                                        for forward/backward difference, the last/first frame is the
                                        frame next to it
                                
                """
            def compute_translational_speed(self, forward: bool):
                """
                MotionData.compute_translational_speed(self, forward: bool)
                 compute the `local` translational velocities of every joint using finite difference
                
                                    note that different from `compute_linear_velocity`, this is the relative
                                    speed of joints wrt. their parents, and the values are represented in the
                                    parents' local coordinates
                
                                    return: an array of size (num_frame, num_joint, 3),
                                        for forward/backward difference, the last/first frame is the
                                        frame next to it
                                
                """
            def flip(self, plane_of_symmetry_normal: typing.Union[typing.List[float], numpy.ndarray], mirror_joint_indices: typing.Optional[typing.List[int]] = None):
                """
                MotionData.flip(self, plane_of_symmetry_normal: Union[List[float], np.ndarray], mirror_joint_indices: Union[None, List[int]] = None)
                 flip the animation wrt the plane of symmetry while assuming the plane passes the origin point
                
                                Note that if the character is not symmetric or if a wrong normal vector is given, the result will not look good
                
                                Parameters
                                ----------
                                plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
                                    note that the
                
                                mirror_joint_indices: should be the index of the mirror joint of a joint
                                            if not provided, get_mirror_joint_indices() will be called to get a best estimation
                
                
                                Returns
                                -------
                                None
                                
                """
            def get_adj_matrix(self) -> ndarray:
                """
                MotionData.get_adj_matrix(self) -> np.ndarray
                """
            def get_end_flags(self) -> ndarray:
                """
                MotionData.get_end_flags(self) -> np.ndarray
                """
            def get_hierarchy(self, copy: bool = False):
                """
                MotionData.get_hierarchy(self, copy: bool = False)
                
                                Get bvh hierarchy
                                
                """
            def get_mirror_joint_indices(self):
                """
                MotionData.get_mirror_joint_indices(self)
                """
            def get_neighbours(self) -> typing.List[typing.List[int]]:
                """
                MotionData.get_neighbours(self) -> List[List[int]]
                """
            def get_reference_pose(self):
                """
                MotionData.get_reference_pose(self)
                """
            def get_t_pose(self):
                """
                MotionData.get_t_pose(self)
                """
            def re_root(self, new_root):
                """
                MotionData.re_root(self, new_root)
                 change the root to another joint
                
                                    the joints will be reordered to ensure that a joint always behind its parent
                                
                """
            def recompute_joint_global_info(self):
                """
                MotionData.recompute_joint_global_info(self)
                """
            def reconfig_reference_pose(self, rotations: typing.Union[typing.List[numpy.ndarray], numpy.ndarray, typing.Dict[str, numpy.ndarray]], treat_as_global_orientations: bool, treat_as_reverse_rotation: bool):
                """
                MotionData.reconfig_reference_pose(self, rotations: Union[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]], treat_as_global_orientations: bool, treat_as_reverse_rotation: bool)
                 reconfigurate the reference pose (T pose) of this bvh object
                                Parameters
                                -------
                                rotations: rotations on the current T pose
                
                                treat_as_global_orientations: if true, the input rotations will be treat as
                                    target orientations of the bones
                
                                treat_as_reverse_rotation: if true, the input rotations are considered as those
                                    rotating the target pose to the current pose
                                
                """
            def remore_reference_nodes(self, new_root):
                """
                MotionData.remore_reference_nodes(self, new_root)
                 create a new skeleton with the root joint as specified
                
                                    some software may export motions with 'reference node', this function will remove those node and bake the
                                    corresponding transformations into the new root
                
                                    note that we only allows a single root joint, so that the siblings of the new_root will be removed
                                
                """
            def remove_end_sites(self, copy: bool = True):
                """
                MotionData.remove_end_sites(self, copy: bool = True)
                """
            def remove_root_pos(self):
                """
                MotionData.remove_root_pos(self)
                
                                Note: this method is in place
                                
                """
            def resample(self, new_fps: int):
                """
                MotionData.resample(self, new_fps: int)
                """
            def reset_global_info(self):
                """
                MotionData.reset_global_info(self)
                """
            def retarget(self, joint_map: typing.Dict[str, typing.Union[str, typing.List[str]]]):
                """
                MotionData.retarget(self, joint_map: Dict[str, Union[str, List[str]]])
                 create a new skeleton based on the joint map and retarget the motion to it
                
                                the hierarchy of current skeleton will be maintained.
                
                                
                """
            def scale(self, factor: float):
                """
                MotionData.scale(self, double factor: float)
                """
            def set_anim_attrs(self, num_frames, fps):
                """
                MotionData.set_anim_attrs(self, num_frames, fps)
                """
            def sub_sequence(self, start: typing.Optional[int] = None, end: typing.Optional[int] = None, skip: typing.Optional[int] = None, copy: bool = True):
                """
                MotionData.sub_sequence(self, start: Optional[int] = None, end: Optional[int] = None, skip: Optional[int] = None, copy: bool = True)
                """
            def symmetrize_skeleton(self, plane_of_symmetry_normal: typing.Union[typing.List[float], numpy.ndarray], mirror_joint_indices: typing.Optional[typing.List[int]]):
                """
                MotionData.symmetrize_skeleton(self, plane_of_symmetry_normal: Union[List[float], np.ndarray], mirror_joint_indices: Union[None, List[int]])
                 fix skeleton joint offsets to make the skeleton symmetric
                
                                Parameters
                                ----------
                                plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
                                    note that the
                
                                mirror_joint_indices: should be the index of the mirror joint of a joint
                                            if not provided, get_mirror_joint_indices() will be called to get a best estimation
                
                                
                """
            def to_contiguous(self):
                """
                MotionData.to_contiguous(self)
                """
            def to_facing_coordinate(self):
                """
                MotionData.to_facing_coordinate(self)
                
                                Note: this method is in place
                                
                """
            def to_local_coordinate(self):
                """
                MotionData.to_local_coordinate(self)
                
                                Note: this method is in place operation
                                
                """
            def z_up_to_y_up(self):
                """
                MotionData.z_up_to_y_up(self)
                """
            @property
            def end_sites(self) -> typing.Optional[typing.List[int]]:
                """
                MotionData.end_sites(self) -> Optional[List[int]]
                """
            @property
            def fps(self):
                """
                MotionData.fps(self)
                """
            @property
            def joint_names(self) -> typing.Optional[typing.List[str]]:
                """
                MotionData.joint_names(self) -> Optional[List[str]]
                """
            @property
            def joint_offsets(self):
                """
                MotionData.joint_offsets(self)
                """
            @property
            def joint_orientation(self) -> typing.Optional[numpy.ndarray]:
                """
                MotionData.joint_orientation(self) -> Optional[np.ndarray]
                """
            @property
            def joint_parents_idx(self):
                """
                MotionData.joint_parents_idx(self)
                """
            @property
            def joint_position(self) -> typing.Optional[numpy.ndarray]:
                """
                MotionData.joint_position(self) -> Optional[np.ndarray]
                """
            @property
            def joint_rotation(self) -> typing.Optional[numpy.ndarray]:
                """
                MotionData.joint_rotation(self) -> Optional[np.ndarray]
                """
            @property
            def joint_translation(self) -> typing.Optional[numpy.ndarray]:
                """
                MotionData.joint_translation(self) -> Optional[np.ndarray]
                """
            @property
            def num_frames(self) -> int:
                """
                MotionData.num_frames(self) -> int
                """
            @property
            def num_joints(self) -> int:
                """
                MotionData.num_joints(self) -> int
                """
    class MotionHelper:
        @staticmethod
        def adjust_root_height(data, dh: typing.Optional[float] = None):
            """
            MotionHelper.adjust_root_height(data, dh: Optional[float] = None)
            """
        @staticmethod
        def calc_children(data):
            """
            MotionHelper.calc_children(data)
            """
        @staticmethod
        def calc_name_idx(data) -> typing.Dict[str, int]:
            """
            MotionHelper.calc_name_idx(data) -> Dict[str, int]
            """
    class Utils:
        @staticmethod
        def align_angles(a: ndarray, degrees: bool, inplace: bool):
            """
            Utils.align_angles(ndarray a: np.ndarray, degrees: bool, inplace: bool)
             make the angles in the array continuous
            
                            we assume the first dim of a is the time
                        
            """
        @staticmethod
        def align_quaternion(qt: ndarray, inplace: bool):
            """
            Utils.align_quaternion(ndarray qt: np.ndarray, inplace: bool)
             make q_n and q_n+1 in the same semisphere
            
                            the first axis of qt should be the time
                        
            """
        @staticmethod
        def animate_motion_data(data, show_skeleton = True, show_animation = True, interval = 1):
            """
            Utils.animate_motion_data(data, show_skeleton=True, show_animation=True, interval=1)
            """
        @staticmethod
        def flip_quaternion(qt: ndarray, normal: ndarray, inplace: bool):
            """
            Utils.flip_quaternion(ndarray qt: np.ndarray, ndarray normal: np.ndarray, inplace: bool)
            """
        @staticmethod
        def flip_vector(vt: ndarray, normal: ndarray, inplace: bool):
            """
            Utils.flip_vector(ndarray vt: np.ndarray, ndarray normal: np.ndarray, inplace: bool)
            """
        @staticmethod
        def get_joint_color(names, left = 'r', right = 'b', otherwise = 'y'):
            """
            Utils.get_joint_color(names, left=u'r', right=u'b', otherwise=u'y')
            """
        @staticmethod
        def quat_product(p: ndarray, q: ndarray, inv_p: bool = False, inv_q: bool = False):
            """
            Utils.quat_product(ndarray p: np.ndarray, ndarray q: np.ndarray, inv_p: bool = False, inv_q: bool = False)
            """
def CloseODE():
    """
    CloseODE()
    
        Deallocate some extra memory used by ODE that can not be deallocated
        using the normal destroy functions.
    """
def InitODE():
    """
    InitODE()
    
        Initialize some ODE internals. This will be called for you when you
        "import ode", but you should call this again if you CloseODE().
    """
def SetInitSeed(*args, **kwargs):
    ...
def Space(*args, **kwargs) -> SpaceBase:
    """
    Space factory function.
    
        Depending on the type argument this function either returns a
        SimpleSpace (space_type=0) or a HashSpace (space_type=1).
    
        This function is provided to remain compatible with previous
        versions of PyODE where there was only one Space class.
    
        >>> space = Space(space_type=0)   # Create a SimpleSpace
        >>> space = Space(space_type=1)   # Create a HashSpace
    """
def __pyx_unpickle_GeomTypes(*args, **kwargs):
    ...
def __pyx_unpickle_JointParam(*args, **kwargs):
    ...
def __pyx_unpickle_JointTypes(*args, **kwargs):
    ...
def areConnected(*args, **kwargs) -> bool:
    """
    areConnected(body1, body2) -> bool
    
        Return True if the two bodies are connected together by a joint,
        otherwise return False.
    
        @param body1: First body
        @type body1: Body
        @param body2: Second body
        @type body2: Body
        @returns: True if the bodies are connected
    """
def collide(*args, **kwargs) -> list:
    """
    collide(geom1, geom2) -> contacts
    
        Generate contact information for two objects.
    
        Given two geometry objects that potentially touch (geom1 and geom2),
        generate contact information for them. Internally, this just calls
        the correct class-specific collision functions for geom1 and geom2.
    
        [flags specifies how contacts should be generated if the objects
        touch. Currently the lower 16 bits of flags specifies the maximum
        number of contact points to generate. If this number is zero, this
        function just pretends that it is one - in other words you can not
        ask for zero contacts. All other bits in flags must be zero. In
        the future the other bits may be used to select other contact
        generation strategies.]
    
        If the objects touch, this returns a list of Contact objects,
        otherwise it returns an empty list.
    
        @param geom1: First Geom
        @type geom1: GeomObject
        @param geom2: Second Geom
        @type geom2: GeomObject
        @returns: Returns a list of Contact objects.
    """
def collide2(*args, **kwargs):
    """
    collide2(geom1, geom2, arg, callback)
    
        Calls the callback for all potentially intersecting pairs that contain
        one geom from geom1 and one geom from geom2.
    
        @param geom1: First Geom
        @type geom1: GeomObject
        @param geom2: Second Geom
        @type geom2: GeomObject
        @param arg: A user argument that is passed to the callback function
        @param callback: Callback function
        @type callback: callable
    """
def decompose_rotation_fast(*args, **kwargs):
    ...
def decompose_rotation_pair_one2many_fast(*args, **kwargs):
    ...
def decompose_rotation_single_fast(*args, **kwargs):
    ...
def decompose_rotation_single_pair_fast(*args, **kwargs):
    ...
def flip_quat_by_w_fast(*args, **kwargs):
    ...
def mix_quat_by_slerp(*args, **kwargs):
    ...
def parent_child_quat_to_hinge_angle_fast(*args, **kwargs):
    ...
def quat_apply_forward_fast(*args, **kwargs):
    ...
def quat_apply_forward_one2many_fast(*args, **kwargs):
    ...
def quat_apply_single_fast(*args, **kwargs):
    ...
def quat_from_matrix_fast(*args, **kwargs):
    ...
def quat_from_matrix_single_fast(*args, **kwargs):
    ...
def quat_from_rotvec_fast(*args, **kwargs):
    ...
def quat_from_rotvec_single_fast(*args, **kwargs):
    ...
def quat_integrate_fast(*args, **kwargs):
    ...
def quat_inv_fast(*args, **kwargs):
    ...
def quat_inv_single_fast(*args, **kwargs):
    ...
def quat_multiply_forward_fast(*args, **kwargs):
    ...
def quat_multiply_forward_one2many_fast(*args, **kwargs):
    ...
def quat_multiply_forward_single(*args, **kwargs):
    ...
def quat_to_hinge_angle_fast(*args, **kwargs):
    ...
def quat_to_matrix_fast(*args, **kwargs):
    ...
def quat_to_matrix_single_fast(*args, **kwargs):
    ...
def quat_to_rotvec_fast(*args, **kwargs):
    ...
def quat_to_rotvec_fast2(*args, **kwargs):
    ...
def quat_to_rotvec_single_fast(*args, **kwargs):
    ...
def quat_to_vec6d_fast(*args, **kwargs):
    ...
def quat_to_vec6d_single_fast(*args, **kwargs):
    ...
def simple_mix_quaternion(*args, **kwargs):
    ...
def six_dim_mat_to_quat_fast(*args, **kwargs):
    ...
def solve_lcp(*args, **kwargs):
    """
        Solve LCP problem.
        given (A,b,lo,hi), solve the LCP problem: A*x = b+w, where each x(i),w(i) satisfies one of
    	(1) x = lo, w >= 0
    	(2) x = hi, w <= 0
    	(3) lo < x < hi, w = 0
        A is a matrix of dimension n*n, everything else is a vector of size n*1.
        lo and hi can be +/- dInfinity as needed. the first `nub' variables are
        unbounded, i.e. hi and lo are assumed to be +/- dInfinity.
    
        we restrict lo(i) <= 0 and hi(i) >= 0.
    
    
        @param a:
        @type a: np.ndarray
        @param b:
        @type b: np.ndarray
        @param nub:
        @type nub: int
        @param lo:
        @type lo: np.ndarray
        @param hi:
        @type hi: np.ndarray
        @param findex:
        @type findex: np.ndarray
        @returns: x, outer_w
    """
def surface_distance_capsule(*args, **kwargs):
    ...
def test_py_list_str_to_std_vector_str(*args, **kwargs):
    ...
def vector_normalize_single_fast(*args, **kwargs):
    ...
def vector_to_cross_matrix_fast(*args, **kwargs):
    ...
def visDrawBackground(x):
    ...
def visDrawWorld():
    ...
def visEndRecordVideo():
    ...
def visGetScreenBuffer():
    ...
def visGetWorld():
    ...
def visKill():
    ...
def visLookAt(pos, target, up):
    ...
def visPause(time):
    ...
def visSetAxisLength(x):
    ...
def visSetColor(col):
    ...
def visSetJointRadius(x):
    ...
def visSetWorld(*args, **kwargs):
    ...
def visStartRecordVideo():
    ...
def visTrackBody(*args, **kwargs):
    ...
def visWhetherHingeAxis(x):
    ...
def visWhetherLocalAxis(x):
    ...
AMotorEuler: int = 1
AMotorUser: int = 0
ContactApprox1: int = 12288
ContactSlip1: int = 256
ContactSlip2: int = 512
Infinity: float  # value = inf
__test__: dict = {'Space (line 7168)': 'Space factory function.\n\n    Depending on the type argument this function either returns a\n    SimpleSpace (space_type=0) or a HashSpace (space_type=1).\n\n    This function is provided to remain compatible with previous\n    versions of PyODE where there was only one Space class.\n\n    >>> space = Space(space_type=0)   # Create a SimpleSpace\n    >>> space = Space(space_type=1)   # Create a HashSpace\n    '}
debug_song = None
environment = None
GeomCCylinder = GeomCapsule
