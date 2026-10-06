// Local modification: per-joint opt-in predictive limits; default behavior is unchanged.
use crate::dynamics::solver::GenericJointConstraint;
use crate::dynamics::{
    FixedJointBuilder, GenericJoint, IntegrationParameters, Multibody, MultibodyLink,
    RigidBodyVelocity, joint,
};
use crate::math::{
    ANG_DIM, DIM, DVector, JacobianViewMut, Pose, Real, Rotation, SPATIAL_DIM, SpatialVector,
    Vector,
};
use parry::math::VectorExt;

#[cfg(feature = "dim2")]
use crate::math::rotation_from_angle;
#[cfg(feature = "dim3")]
use crate::utils::RotationOps;
use crate::utils::vect_to_na;
use na::DVectorViewMut;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
/// An joint attached to two bodies based on the reduced coordinates formalism.
pub struct MultibodyJoint {
    /// The joint’s description.
    pub data: GenericJoint,
    /// Is the joint a kinematic joint?
    ///
    /// Kinematic joint velocities are never changed by the physics engine. This gives the user
    /// total control over the values of their degrees of freedoms.
    pub kinematic: bool,
    pub(crate) coords: SpatialVector,
    pub(crate) joint_rot: Rotation,
    /// Per-DoF spring stiffness (a passive joint spring integrated implicitly
    /// in the generalized dynamics). Zero on axes with no spring.
    pub(crate) spring_stiffness: SpatialVector,
    /// Per-DoF spring rest position, in the joint's generalized coordinate
    /// (same convention as [`Self::coords`]). Only meaningful where
    /// `spring_stiffness` is non-zero.
    pub(crate) spring_ref: SpatialVector,
    /// Explicit opt-in to two unilateral velocity bounds at the actual stops.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    predictive_limits: bool,
    /// Independent dimensionless dry-friction compliance, without changing
    /// limit or locked-axis softness. None preserves the original coefficients.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    dry_friction_cfm_coefficient: Option<Real>,
    /// Explicitly selected, single-axis source-limit experiment.
    #[cfg(feature = "sim2sim-source-limit-probe")]
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    sim2sim_source_limit_probe: Option<joint::SourceLimitProbe>,
}


impl MultibodyJoint {
    /// Creates a new multibody joint from its description.
    pub fn new(data: GenericJoint, kinematic: bool) -> Self {
        Self {
            data,
            kinematic,
            coords: Default::default(),
            joint_rot: Rotation::IDENTITY,
            spring_stiffness: Default::default(),
            spring_ref: Default::default(),
            predictive_limits: false,
            dry_friction_cfm_coefficient: None,
            #[cfg(feature = "sim2sim-source-limit-probe")]
            sim2sim_source_limit_probe: None,
        }
    }

    /// Sets a passive joint spring on `axis` (an index into the 6-DoF spatial
    /// layout: `0..DIM` linear, `DIM..SPATIAL_DIM` angular). The spring applies
    /// a generalized force `-stiffness · (q − rest)` integrated implicitly.
    pub fn set_spring(&mut self, axis: usize, stiffness: Real, rest: Real) {
        self.spring_stiffness[axis] = stiffness;
        self.spring_ref[axis] = rest;
    }

    /// The passive joint spring on `axis` as `(stiffness, rest)`, as set by
    /// [`Self::set_spring`]. `(0, 0)` if no spring is set on that axis.
    pub fn spring(&self, axis: usize) -> (Real, Real) {
        (self.spring_stiffness[axis], self.spring_ref[axis])
    }

    /// Select native rows enforcing `min <= q + dt*v <= max` on a scalar hinge or slide.
    /// This does not move either stop, overwrite state, or subdivide time.
    /// Returns false for unsupported/kinematic joints, invalid finite stops,
    /// or a conflicting source-limit diagnostic.
    pub fn set_predictive_limits_enabled(&mut self, enabled: bool) -> bool {
        if enabled {
            let free_limits = self.data.limit_axes.bits() & !self.data.locked_axes.bits();
            if self.kinematic
                || self.ndofs() != 1
                || free_limits == 0
                || (0..SPATIAL_DIM).any(|axis| {
                    free_limits & (1 << axis) != 0
                        && (!self.data.limits[axis].min.is_finite()
                            || !self.data.limits[axis].max.is_finite()
                            || self.data.limits[axis].min >= self.data.limits[axis].max)
                })
            {
                return false;
            }
        }
        #[cfg(feature = "sim2sim-source-limit-probe")]
        if enabled && self.sim2sim_source_limit_probe.is_some() {
            return false;
        }
        self.predictive_limits = enabled;
        true
    }

    /// Whether this joint explicitly selected the predictive native rows.
    pub fn predictive_limits_enabled(&self) -> bool {
        self.predictive_limits
    }

    /// Select a dimensionless dry-friction CFM coefficient independently of
    /// joint-limit softness. The existing friction row multiplies it by its
    /// actual inverse effective mass. Peak friction and constraint bounds stay
    /// unchanged. None restores the original joint-softness-derived value.
    /// Returns false for non-finite/negative coefficients or non-scalar joints.
    pub fn set_dry_friction_cfm_coefficient(&mut self, coefficient: Option<Real>) -> bool {
        if self.kinematic
            || self.ndofs() != 1
            || coefficient.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return false;
        }
        self.dry_friction_cfm_coefficient = coefficient;
        true
    }

    /// Explicit dry-friction coefficient, independent of joint-limit softness.
    pub fn dry_friction_cfm_coefficient(&self) -> Option<Real> {
        self.dry_friction_cfm_coefficient
    }

    /// Select the MuJoCo-style AngX limit-row experiment on this joint only.
    #[cfg(feature = "sim2sim-source-limit-probe")]
    pub fn sim2sim_set_source_limit_probe(
        &mut self,
        solref: [Real; 2],
        solimp: [Real; 5],
        margin: Real,
        dof_invweight0: Real,
    ) -> bool {
        let valid = !self.predictive_limits
            && solref.iter().chain(solimp.iter()).all(|v| v.is_finite())
            && margin == 0.0
            && solref[0] > 0.0
            && solref[1] > 0.0
            && 0.0 < solimp[0]
            && solimp[0] <= solimp[1]
            && solimp[1] < 1.0
            && solimp[2] > 0.0
            && solimp[3] == 0.5
            && solimp[4] == 2.0
            && dof_invweight0.is_finite()
            && dof_invweight0 > 0.0
            && self
                .data
                .limit_axes
                .contains(crate::dynamics::JointAxis::AngX.into());
        if valid {
            self.sim2sim_source_limit_probe = Some(joint::SourceLimitProbe {
                solref,
                solimp,
                margin,
                dof_invweight0,
            });
        }
        valid
    }

    pub(crate) fn free(pos: Pose) -> Self {
        let mut result = Self::new(GenericJoint::default(), false);
        result.set_free_pos(pos);
        result
    }

    pub(crate) fn fixed(pos: Pose) -> Self {
        Self::new(
            FixedJointBuilder::new().local_frame1(pos).build().into(),
            false,
        )
    }

    pub(crate) fn set_free_pos(&mut self, pos: Pose) {
        #[cfg(feature = "dim2")]
        {
            self.coords.x = pos.translation.x;
            self.coords.y = pos.translation.y;
        }
        #[cfg(feature = "dim3")]
        {
            self.coords[0] = pos.translation.x;
            self.coords[1] = pos.translation.y;
            self.coords[2] = pos.translation.z;
        }
        self.joint_rot = pos.rotation;
    }

    /// The joint’s angular coordinates converted to a rotation.
    pub fn joint_rot(&self) -> Rotation {
        self.joint_rot
    }

    fn num_free_lin_dofs(&self) -> usize {
        let locked_bits = self.data.locked_axes.bits();
        DIM - (locked_bits & ((1 << DIM) - 1)).count_ones() as usize
    }

    /// Generalized coordinates for this joint.
    pub fn coords(&self) -> SpatialVector {
        self.coords
    }

    /// The number of degrees of freedom allowed by the multibody_joint.
    pub fn ndofs(&self) -> usize {
        SPATIAL_DIM - self.data.locked_axes.bits().count_ones() as usize
    }

    /// The position of the multibody link containing this multibody_joint relative to its parent.
    pub fn body_to_parent(&self) -> Pose {
        let locked_bits = self.data.locked_axes.bits();
        let mut transform = Pose::from_rotation(self.joint_rot) * self.data.local_frame2.inverse();

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                // Create a translation along axis i with the coordinate value
                let translation = Vector::ith(i, self.coords[i]);
                transform = Pose::from_translation(translation) * transform;
            }
        }

        self.data.local_frame1 * transform
    }

    /// Integrate the position of this multibody_joint.
    #[profiling::function]
    pub fn integrate(&mut self, dt: Real, vels: &[Real]) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                self.coords[i] += vels[curr_free_dof] * dt;
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                self.coords[DIM + dof_id] += vels[curr_free_dof] * dt;
                #[cfg(feature = "dim2")]
                {
                    self.joint_rot = rotation_from_angle(self.coords[DIM + dof_id]);
                }
                #[cfg(feature = "dim3")]
                {
                    self.joint_rot = Rotation::from_axis_angle(
                        Vector::ith(dof_id, 1.0),
                        self.coords[DIM + dof_id],
                    );
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let angvel = Vector::from_slice(&vels[curr_free_dof..curr_free_dof + 3]);
                let disp = Rotation::from_scaled_axis(angvel * dt);
                // Rotation is a unit quaternion. glam multiplication does not
                // renormalize; repeated f32 products otherwise drift off SO(3).
                // Keep the invariant inside the existing integration, without
                // adding a step or an external pose/velocity overwrite.
                self.joint_rot = (disp * self.joint_rot).normalize();
                self.coords[3] += angvel[0] * dt;
                self.coords[4] += angvel[1] * dt;
                self.coords[5] += angvel[2] * dt;
            }
            _ => unreachable!(),
        }
    }

    /// Apply a displacement to the multibody_joint.
    pub fn apply_displacement(&mut self, disp: &[Real]) {
        self.integrate(1.0, disp);
    }

    /// Sets in `out` the non-zero entries of the multibody_joint jacobian transformed by `transform`.
    pub fn jacobian(&self, transform: &Rotation, out: &mut JacobianViewMut<Real>) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                let transformed_axis = (*transform) * Vector::ith(i, 1.0);
                out.fixed_view_mut::<DIM, 1>(0, curr_free_dof)
                    .copy_from(&vect_to_na(transformed_axis));
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                #[cfg(feature = "dim2")]
                {
                    out[(DIM, curr_free_dof)] = 1.0;
                }

                #[cfg(feature = "dim3")]
                {
                    let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                    let rotmat = transform.to_mat();
                    out.fixed_view_mut::<ANG_DIM, 1>(DIM, curr_free_dof)
                        .copy_from_slice(rotmat.col(dof_id).as_ref());
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let rotmat = transform.to_mat();
                out.fixed_view_mut::<3, 3>(3, curr_free_dof)
                    .copy_from_slice(rotmat.as_ref());
            }
            _ => unreachable!(),
        }
    }

    /// Multiply the multibody_joint jacobian by generalized velocities to obtain the
    /// relative velocity of the multibody link containing this multibody_joint.
    pub fn jacobian_mul_coordinates(&self, acc: &[Real]) -> RigidBodyVelocity<Real> {
        let locked_bits = self.data.locked_axes.bits();
        let mut result = RigidBodyVelocity::zero();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                result.linvel += Vector::ith(i, acc[curr_free_dof]);
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                #[cfg(feature = "dim2")]
                {
                    result.angvel += acc[curr_free_dof];
                }
                #[cfg(feature = "dim3")]
                {
                    let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                    result.angvel[dof_id] += acc[curr_free_dof];
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let angvel = Vector::from_slice(&acc[curr_free_dof..curr_free_dof + 3]);
                result.angvel += angvel;
            }
            _ => unreachable!(),
        }
        result
    }

    /// Fill `out` with the non-zero entries of a damping that can be applied by default to ensure a good stability of the multibody_joint.
    pub fn default_damping(&self, out: &mut DVectorViewMut<Real>) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = self.num_free_lin_dofs();

        // A default damping only for the angular dofs
        for i in DIM..SPATIAL_DIM {
            if locked_bits & (1 << i) == 0 {
                // This is a free angular DOF.
                out[curr_free_dof] = 0.1;
                curr_free_dof += 1;
            }
        }
    }

    /// Maximum number of velocity constrains that can be generated by this multibody_joint.
    pub fn num_velocity_constraints(&self) -> usize {
        let locked_bits = self.data.locked_axes.bits();
        let limit_bits = self.data.limit_axes.bits();
        let motor_bits = self.data.motor_axes.bits();
        let mut num_constraints = 0;

        for i in 0..SPATIAL_DIM {
            if (locked_bits & (1 << i)) == 0 {
                if (limit_bits & (1 << i)) != 0 {
                    num_constraints += if self.predictive_limits { 2 } else { 1 };
                }
                if (motor_bits & (1 << i)) != 0 {
                    num_constraints += 1;
                }
            }
        }

        num_constraints
    }

    /// Initialize and generate velocity constraints to enforce, e.g., multibody_joint limits and motors.
    pub fn velocity_constraints(
        &self,
        params: &IntegrationParameters,
        multibody: &Multibody,
        link: &MultibodyLink,
        mut j_id: usize,
        jacobians: &mut DVector,
        constraints: &mut [GenericJointConstraint],
    ) -> usize {
        let j_id = &mut j_id;
        let locked_bits = self.data.locked_axes.bits();
        let limit_bits = self.data.limit_axes.bits();
        let motor_bits = self.data.motor_axes.bits();
        let mut num_constraints = 0;
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                let limits = if (limit_bits & (1 << i)) != 0 {
                    Some([self.data.limits[i].min, self.data.limits[i].max])
                } else {
                    None
                };

                if (motor_bits & (1 << i)) != 0 {
                    joint::unit_joint_motor_constraint(
                        params,
                        multibody,
                        link,
                        &self.data.motors[i],
                        self.coords[i],
                        limits,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                    );
                }

                if (limit_bits & (1 << i)) != 0 {
                    joint::unit_joint_limit_constraint(
                        params,
                        multibody,
                        link,
                        [self.data.limits[i].min, self.data.limits[i].max],
                        self.coords[i],
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                }
                // Dry joint friction (MuJoCo `frictionloss`), keyed per-DoF on
                // the multibody rather than per-axis on the joint. Zero (the
                // default) emits nothing.
                let friction = multibody.frictions()[link.assembly_id + curr_free_dof];
                if friction > 0.0 {
                    joint::unit_joint_friction_constraint(
                        params,
                        multibody,
                        link,
                        friction,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                        self.dry_friction_cfm_coefficient,
                    );
                }
                curr_free_dof += 1;
            }
        }

        /*
        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {}
            2 => {
                todo!()
            }
            3 => {}
            _ => unreachable!(),
        }
         */
        // TODO: we should make special cases for multi-angular-dofs limits/motors
        for i in DIM..SPATIAL_DIM {
            if (locked_bits & (1 << i)) == 0 {
                let limits = if (limit_bits & (1 << i)) != 0 {
                    let limits = [self.data.limits[i].min, self.data.limits[i].max];
                    #[cfg(feature = "sim2sim-source-limit-probe")]
                    if i == DIM && self.sim2sim_source_limit_probe.is_some() {
                        joint::unit_joint_source_limit_probe_constraint(
                            params,
                            multibody,
                            link,
                            limits,
                            self.coords[i],
                            curr_free_dof,
                            j_id,
                            jacobians,
                            constraints,
                            &mut num_constraints,
                            self.sim2sim_source_limit_probe.as_ref().unwrap(),
                        );
                    } else {
                        joint::unit_joint_limit_constraint(
                            params,
                            multibody,
                            link,
                            limits,
                            self.coords[i],
                            curr_free_dof,
                            j_id,
                            jacobians,
                            constraints,
                            &mut num_constraints,
                            self.data.softness,
                        );
                    }
                    #[cfg(not(feature = "sim2sim-source-limit-probe"))]
                    joint::unit_joint_limit_constraint(
                        params,
                        multibody,
                        link,
                        limits,
                        self.coords[i],
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                    Some(limits)
                } else {
                    None
                };

                if (motor_bits & (1 << i)) != 0 {
                    joint::unit_joint_motor_constraint(
                        params,
                        multibody,
                        link,
                        &self.data.motors[i],
                        self.coords[i],
                        limits,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                    );
                }

                // Dry joint friction (MuJoCo `frictionloss`), keyed per-DoF on
                // the multibody rather than per-axis on the joint. Zero (the
                // default) emits nothing.
                let friction = multibody.frictions()[link.assembly_id + curr_free_dof];
                if friction > 0.0 {
                    joint::unit_joint_friction_constraint(
                        params,
                        multibody,
                        link,
                        friction,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                        self.dry_friction_cfm_coefficient,
                    );
                }
                curr_free_dof += 1;
            }
        }

        num_constraints
    }
}
