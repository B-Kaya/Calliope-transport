�HDF

         ��������p�     0       ʾ�EOHDR�	"     0       �      �     �     
          �	      z
      �
       T	                                                                                                                                                                                                                                               O                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  �E#FRHP                    -      �
       @              0             +�                                            (  �       wD)�BTHD       d(              �?��BTHD 	      d(              C6FSHD  �	                            P x (        ��      D       D       �%�"BTLF  {    <     �� T&   @     �*% O/   E     l3�- /   1     +�
3 H    3     ��!6    �    �Q@     >    kW�G     2      @�k �   9     ��w (       �[S� �    (     ʻ#� �   &  
   ��� �   .     Ŵ�� �   2     �]Y� !   r     f��       	   Y�,� �    "     ��N�f���                                                                                                                                                                                                                     BTLF 	     2       H    3      {    <      �    (      �    "          >     T&   @         �     �   9            	    �   &  
    (        !   r      �   2      �   .      /   1      O/   E     ks2                                                                                                                                                                                                                                                                                         BTHD       d(i�              ��k�     math    �     constraints:
  flow_capacity_per_storage_capacity_min:
    description: Set the lower bound of storage flow capacity relative to its storage
      capacity.
    foreach:
    - nodes
    - techs
    - carriers
    where: storage_cap AND flow_cap_per_storage_cap_min
    equations:
    - expression: flow_cap >= storage_cap * flow_cap_per_storage_cap_min
  flow_capacity_per_storage_capacity_max:
    description: Set the upper bound of storage flow capacity relative to its storage
      capacity.
    foreach:
    - nodes
    - techs
    - carriers
    where: storage_cap AND flow_cap_per_storage_cap_max
    equations:
    - expression: flow_cap <= storage_cap * flow_cap_per_storage_cap_max
  source_capacity_equals_flow_capacity:
    description: Set a `supply` technology's flow capacity to equal its source capacity.
    foreach:
    - nodes
    - techs
    - carriers
    where: source_cap AND source_cap_equals_flow_cap=True
    equations:
    - expression: source_cap == flow_cap
  force_zero_area_use:
    description: Set a technology's area use to zero if its flow capacity upper bound
      is zero.
    foreach:
    - nodes
    - techs
    where: area_use AND flow_cap_max=0
    equations:
    - expression: area_use == 0
  area_use_per_flow_capacity:
    description: Set a fixed relationship between a technology's flow capacity and
      its area use.
    foreach:
    - nodes
    - techs
    - carriers
    where: area_use AND area_use_per_flow_cap
    equations:
    - expression: area_use == flow_cap * area_use_per_flow_cap
  area_use_capacity_per_loc:
    description: Set an upper bound on the total area that all technologies with `area_use`
      can occupy at a given node.
    foreach:
    - nodes
    where: area_use AND available_area
    equations:
    - expression: sum(area_use, over=techs) <= available_area
  flow_capacity_systemwide_max:
    description: Set an upper bound on flow capacity of a technology across all nodes
      in which the technology exists.
    foreach:
    - techs
    - carriers
    where: flow_cap_max_systemwide
    equations:
    - expression: sum(flow_cap, over=nodes) <= flow_cap_max_systemwide
  flow_capacity_systemwide_min:
    description: Set a lower bound on flow capacity of a technology across all nodes
      in which the technology exists.
    foreach:
    - techs
    - carriers
    where: flow_cap_min_systemwide
    equations:
    - expression: sum(flow_cap, over=nodes) >= flow_cap_min_systemwide
  balance_conversion:
    description: Fix the relationship between a `conversion` technology's outflow
      and consumption.
    foreach:
    - nodes
    - techs
    - timesteps
    where: base_tech=conversion AND NOT include_storage=true
    equations:
    - expression: sum(flow_out_inc_eff, over=carriers) == sum(flow_in_inc_eff, over=carriers)
  flow_out_max:
    description: Set the upper bound of a technology's outflow.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: carrier_out AND NOT operating_units
    equations:
    - expression: flow_out <= flow_cap * timestep_resolution * flow_out_parasitic_eff
  flow_out_min:
    description: Set the lower bound of a technology's outflow.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_out_min_relative AND NOT operating_units
    equations:
    - expression: flow_out >= flow_cap * timestep_resolution * flow_out_min_relative
  flow_in_max:
    description: Set the upper bound of a technology's inflow.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: carrier_in AND NOT operating_units
    equations:
    - expression: flow_in <= flow_cap * timestep_resolution
  source_max:
    description: Set the upper bound of a `supply` technology's source consumption.
    foreach:
    - nodes
    - techs
    - timesteps
    where: source_cap
    equations:
    - expression: source_use <= timestep_resolution * source_cap
  storage_max:
    description: Set the upper bound of the amount of carrier a technology can store.
    foreach:
    - nodes
    - techs
    - timesteps
    where: storage
    equations:
    - expression: storage <= storage_cap
  storage_discharge_depth_limit:
    description: Set the lower bound of the stored carrier a technology must keep
      in reserve at all times.
    foreach:
    - nodes
    - techs
    - timesteps
    where: storage AND storage_discharge_depth
    equations:
    - expression: storage - storage_discharge_depth * storage_cap >= 0
  system_balance:
    description: Set the global carrier balance of the optimisation problem by fixing
      the total production of a given carrier to equal the total consumption of that
      carrier at every node in every timestep.
    foreach:
    - nodes
    - carriers
    - timesteps
    equations:
    - expression: sum(flow_out, over=techs) - sum(flow_in, over=techs) - $flow_export
        + $unmet_demand_and_unused_supply == 0
    sub_expressions:
      flow_export:
      - where: any(carrier_export, over=techs)
        expression: sum(flow_export, over=techs)
      - where: NOT any(carrier_export, over=techs)
        expression: '0'
      unmet_demand_and_unused_supply:
      - where: config.ensure_feasibility=True
        expression: unmet_demand + unused_supply
      - where: NOT config.ensure_feasibility=True
        expression: '0'
  balance_demand:
    description: Set the upper bound on, or a fixed total of, that a demand technology
      must dump to its sink in each timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: base_tech=demand
    equations:
    - where: sink_use_equals
      expression: flow_in_inc_eff == sink_use_equals * $sink_scaler
    - where: NOT sink_use_equals AND sink_use_max
      expression: flow_in_inc_eff <= sink_use_max * $sink_scaler
    sub_expressions:
      sink_scaler:
      - where: sink_unit=per_area
        expression: area_use
      - where: sink_unit=per_cap
        expression: sum(flow_cap, over=carriers)
      - where: sink_unit=absolute
        expression: '1'
  balance_demand_min_use:
    description: Set the lower bound on the quantity of flow a `demand` technology
      must dump to its sink in each timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: sink_use_min AND NOT sink_use_equals AND base_tech=demand
    equations:
    - expression: flow_in_inc_eff >= sink_use_min * $sink_scaler
    sub_expressions:
      sink_scaler:
      - where: sink_unit=per_area
        expression: area_use
      - where: sink_unit=per_cap
        expression: sum(flow_cap, over=carriers)
      - where: sink_unit=absolute
        expression: '1'
  balance_supply_no_storage:
    description: Fix the outflow of a `supply` technology to its consumption of the
      available source.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: base_tech=supply AND NOT include_storage=True
    equations:
    - expression: flow_out_inc_eff == source_use * source_eff
  balance_supply_with_storage:
    description: Fix the outflow of a `supply` technology to its consumption of the
      available source, with a storage buffer to temporally offset the outflow from
      source consumption.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: storage AND base_tech=supply
    equations:
    - expression: storage == $storage_previous_step + source_use * source_eff - flow_out_inc_eff
    sub_expressions:
      storage_previous_step:
      - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
        expression: storage_initial * storage_cap
      - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
          \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT lookup_cluster_first_timestep=True"
        expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1) *
          roll(storage, timesteps=1)
      - where: lookup_cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
          AND NOT cyclic_storage=True)
        expression: (1 - storage_loss) ** select_from_lookup_arrays(timestep_resolution,
          timesteps=lookup_cluster_last_timestep) * select_from_lookup_arrays(storage,
          timesteps=lookup_cluster_last_timestep)
  source_availability_supply:
    description: Set the upper bound on, or a fixed total of, a `supply` technology's
      ability to consume its available resource.
    foreach:
    - nodes
    - techs
    - timesteps
    where: source_use AND (source_use_equals OR source_use_max)
    equations:
    - where: source_use_equals
      expression: source_use == source_use_equals * $source_scaler
    - where: NOT source_use_equals AND source_use_max
      expression: source_use <= source_use_max * $source_scaler
    sub_expressions:
      source_scaler:
      - where: source_unit=per_area
        expression: area_use
      - where: source_unit=per_cap
        expression: sum(flow_cap, over=carriers)
      - where: source_unit=absolute
        expression: '1'
  balance_supply_min_use:
    description: Set the lower bound on the quantity of its source a `supply` technology
      must use in each timestep.
    foreach:
    - nodes
    - techs
    - timesteps
    where: source_use_min AND NOT source_use_equals AND base_tech=supply
    equations:
    - expression: source_use >= source_use_min * $source_scaler
    sub_expressions:
      source_scaler:
      - where: source_unit=per_area
        expression: area_use
      - where: source_unit=per_cap
        expression: sum(flow_cap, over=carriers)
      - where: source_unit=absolute
        expression: '1'
  balance_storage:
    description: Fix the quantity of carrier stored in a `storage` technology at the
      end of each timestep based on the net flow of carrier charged and discharged
      and the quantity of carrier stored at the start of the timestep.
    foreach:
    - nodes
    - techs
    - timesteps
    where: (include_storage=true or base_tech=storage) AND NOT (base_tech=supply OR
      base_tech=demand)
    equations:
    - expression: "storage == $storage_previous_step -\n  sum(flow_out_inc_eff, over=carriers)
        + sum(flow_in_inc_eff, over=carriers)"
    sub_expressions:
      storage_previous_step:
      - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
        expression: storage_initial * storage_cap
      - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
          \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT lookup_cluster_first_timestep=True"
        expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1) *
          roll(storage, timesteps=1)
      - where: lookup_cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
          AND NOT cyclic_storage=True)
        expression: (1 - storage_loss) ** select_from_lookup_arrays(timestep_resolution,
          timesteps=lookup_cluster_last_timestep) * select_from_lookup_arrays(storage,
          timesteps=lookup_cluster_last_timestep)
  set_storage_initial:
    description: Fix the relationship between carrier stored in a `storage` technology
      at the start and end of the whole model period.
    foreach:
    - nodes
    - techs
    where: storage AND storage_initial AND cyclic_storage=True
    equations:
    - expression: "storage[timesteps=$final_step] * (\n  (1 - storage_loss) ** timestep_resolution[timesteps=$final_step]\n
        ) == storage_initial * storage_cap"
    slices:
      final_step:
      - expression: get_val_at_index(timesteps=-1)
    active: true
  balance_transmission:
    description: Fix the relationship between between carrier flowing into and out
      of a `transmission` link in each timestep.
    foreach:
    - techs
    - timesteps
    where: base_tech=transmission
    equations:
    - expression: sum(flow_out_inc_eff, over=[nodes, carriers]) == sum(flow_in_inc_eff,
        over=[nodes, carriers])
  symmetric_transmission:
    description: Fix the flow capacity of two `transmission` technologies representing
      the same link in the system.
    foreach:
    - nodes
    - techs
    where: base_tech=transmission
    equations:
    - expression: sum(flow_cap, over=carriers) == link_flow_cap
  export_balance:
    description: Set the lower bound of a technology's outflow to a technology's carrier
      export, for any technologies that can export carriers out of the system.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_export
    equations:
    - expression: flow_out >= flow_export
  flow_export_max:
    description: Set the upper bound of a technology's carrier export, for any technologies
      that can export carriers out of the system.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_export AND export_max
    equations:
    - where: operating_units
      expression: flow_export <= export_max * operating_units
    - where: NOT operating_units
      expression: flow_export <= export_max
  unit_commitment_milp:
    description: Set the upper bound of the number of integer units of technology
      that can exist, for any technology using integer units to define its capacity.
    foreach:
    - nodes
    - techs
    - timesteps
    where: operating_units AND purchased_units
    equations:
    - expression: operating_units <= purchased_units
  available_flow_cap_binary:
    description: Limit flow capacity to zero if the technology is not operating in
      a given timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: available_flow_cap
    equations:
    - expression: available_flow_cap <= flow_cap_max * operating_units
  available_flow_cap_continuous:
    description: Limit flow capacity to the value of the `flow_cap` decision variable
      when the technology is operating in a given timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: available_flow_cap
    equations:
    - expression: available_flow_cap <= flow_cap
  available_flow_cap_max_binary_continuous_switch:
    description: Force flow capacity to equal the value of the `flow_cap` decision
      variable if the technology is operating in a given timestep, zero otherwise.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: available_flow_cap
    equations:
    - expression: available_flow_cap >= flow_cap + ((operating_units - purchased_units)
        * flow_cap_max)
  flow_out_max_milp:
    description: Set the upper bound of a technology's ability to produce carriers,
      for any technology using integer units to define its capacity.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_out AND operating_units AND flow_cap_per_unit
    equations:
    - expression: flow_out <= operating_units * timestep_resolution * flow_cap_per_unit
        * flow_out_parasitic_eff
  flow_in_max_milp:
    description: Set the upper bound of a technology's ability to consume carriers,
      for any technology using integer units to define its capacity.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_in AND operating_units AND flow_cap_per_unit
    equations:
    - expression: flow_in <= operating_units * timestep_resolution * flow_cap_per_unit
  flow_out_min_milp:
    description: Set the lower bound of a technology's ability to produce carriers,
      for any technology using integer units to define its capacity.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_out AND operating_units AND flow_out_min_relative
    equations:
    - where: flow_cap_per_unit
      expression: flow_out >= operating_units * timestep_resolution * flow_cap_per_unit
        * flow_out_min_relative
    - where: available_flow_cap
      expression: flow_out >= available_flow_cap * timestep_resolution * flow_out_min_relative
  storage_capacity_units_milp:
    description: Fix the storage capacity of any technology using integer units to
      define its capacity.
    foreach:
    - nodes
    - techs
    - carriers
    where: storage AND purchased_units AND storage_cap_per_unit
    equations:
    - expression: storage_cap == purchased_units * storage_cap_per_unit
  flow_capacity_units_milp:
    description: Fix the flow capacity of any technology using integer units to define
      its capacity.
    foreach:
    - nodes
    - techs
    - carriers
    where: operating_units AND flow_cap_per_unit
    equations:
    - expression: flow_cap == purchased_units * flow_cap_per_unit
  flow_capacity_max_purchase_milp:
    description: Set the upper bound on a technology's flow capacity, for any technology
      with integer capacity purchasing.
    foreach:
    - nodes
    - techs
    - carriers
    where: purchased_units
    equations:
    - where: flow_cap_max
      expression: flow_cap <= flow_cap_max * purchased_units
    - where: NOT flow_cap_max
      expression: flow_cap <= bigM * purchased_units
  flow_capacity_min_purchase_milp:
    description: Set the lower bound on a technology's flow capacity, for any technology
      with integer capacity purchasing.
    foreach:
    - nodes
    - techs
    - carriers
    where: purchased_units AND flow_cap_min
    equations:
    - expression: flow_cap >= flow_cap_min * purchased_units
  storage_capacity_max_purchase_milp:
    description: Set the upper bound on a technology's storage capacity, for any technology
      with integer capacity purchasing.
    foreach:
    - nodes
    - techs
    where: purchased_units AND storage_cap_max
    equations:
    - expression: storage_cap <= storage_cap_max * purchased_units
  storage_capacity_min_purchase_milp:
    description: Set the lower bound on a technology's storage capacity, for any technology
      with integer capacity purchasing.
    foreach:
    - nodes
    - techs
    where: purchased_units AND storage_cap_min
    equations:
    - expression: storage_cap >= storage_cap_min * purchased_units
  unit_capacity_max_systemwide_milp:
    description: Set the upper bound on the total number of units of a technology
      that can be purchased across all nodes where the technology can exist, for any
      technology using integer units to define its capacity.
    foreach:
    - techs
    where: purchased_units AND purchased_units_max_systemwide
    equations:
    - expression: sum(purchased_units, over=nodes) <= purchased_units_max_systemwide
  unit_capacity_min_systemwide_milp:
    description: Set the lower bound on the total number of units of a technology
      that can be purchased across all nodes where the technology can exist, for any
      technology using integer units to define its capacity.
    foreach:
    - techs
    where: purchased_units AND purchased_units_max_systemwide
    equations:
    - expression: sum(purchased_units, over=nodes) >= purchased_units_min_systemwide
  async_flow_in_milp:
    description: Set a technology's ability to have inflow in the same timestep that
      it has outflow, for any technology using the asynchronous flow binary switch.
    foreach:
    - nodes
    - techs
    - timesteps
    where: async_flow_switch
    equations:
    - expression: sum(flow_in, over=carriers) <= (1 - async_flow_switch) * bigM
  async_flow_out_milp:
    description: Set a technology's ability to have outflow in the same timestep that
      it has inflow, for any technology using the asynchronous flow binary switch.
    foreach:
    - nodes
    - techs
    - timesteps
    where: async_flow_switch
    equations:
    - expression: sum(flow_out, over=carriers) <= async_flow_switch * bigM
  ramping_up:
    description: Set the upper bound on a technology's ability to ramp outflow up
      beyond a certain percentage compared to the previous timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_ramping AND NOT timesteps=get_val_at_index(timesteps=0)
    equations:
    - expression: $flow - roll($flow, timesteps=1) <= flow_ramping * flow_cap
    sub_expressions:
      flow:
      - where: carrier_out AND NOT carrier_in
        expression: flow_out / timestep_resolution
      - where: carrier_in AND NOT carrier_out
        expression: flow_in / timestep_resolution
      - where: carrier_in AND carrier_out
        expression: (flow_out - flow_in) / timestep_resolution
  ramping_down:
    description: Set the upper bound on a technology's ability to ramp outflow down
      beyond a certain percentage compared to the previous timestep.
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_ramping AND NOT timesteps=get_val_at_index(timesteps=0)
    equations:
    - expression: -1 * flow_ramping * flow_cap <= $flow - roll($flow, timesteps=1)
    sub_expressions:
      flow:
      - where: carrier_out AND NOT carrier_in
        expression: flow_out / timestep_resolution
      - where: carrier_in AND NOT carrier_out
        expression: flow_in / timestep_resolution
      - where: carrier_in AND carrier_out
        expression: (flow_out - flow_in) / timestep_resolution
variables:
  flow_cap:
    description: A technology's flow capacity, also known as its nominal or nameplate
      capacity.
    default: 0
    unit: power
    foreach:
    - nodes
    - techs
    - carriers
    bounds:
      min: flow_cap_min
      max: flow_cap_max
  link_flow_cap:
    description: A transmission technology's flow capacity, also known as its nominal
      or nameplate capacity.
    default: 0
    unit: power
    foreach:
    - techs
    where: base_tech=transmission
    bounds:
      min: 0
      max: .inf
  flow_out:
    description: The outflow of a technology per timestep, also known as the flow
      discharged (from `storage` technologies) or the flow received (by `transmission`
      technologies) on a link.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: carrier_out
    bounds:
      min: 0
      max: .inf
  flow_in:
    description: The inflow to a technology per timestep, also known as the flow consumed
      (by `storage` technologies) or the flow sent (by `transmission` technologies)
      on a link.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: carrier_in
    bounds:
      min: 0
      max: .inf
  flow_export:
    description: The flow of a carrier exported outside the system boundaries by a
      technology per timestep.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: carrier_export
    bounds:
      min: 0
      max: .inf
  area_use:
    description: The area in space utilised directly (e.g., solar PV panels) or indirectly
      (e.g., biofuel crops) by a technology.
    default: 0
    unit: area
    foreach:
    - nodes
    - techs
    where: (area_use_min OR area_use_max OR area_use_per_flow_cap OR sink_unit=per_area
      OR source_unit=per_area)
    bounds:
      min: area_use_min
      max: area_use_max
  source_use:
    description: The carrier flow consumed from outside the system boundaries by a
      `supply` technology.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    - timesteps
    where: base_tech=supply
    bounds:
      min: 0
      max: .inf
  source_cap:
    description: The upper limit on a flow that can be consumed from outside the system
      boundaries by a `supply` technology in each timestep.
    default: 0
    unit: power
    foreach:
    - nodes
    - techs
    where: base_tech=supply
    bounds:
      min: source_cap_min
      max: source_cap_max
  storage_cap:
    description: The upper limit on a carrier that can be stored by a technology in
      any timestep.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    where: include_storage=True OR base_tech=storage
    domain: real
    bounds:
      min: storage_cap_min
      max: storage_cap_max
    active: true
  storage:
    description: The carrier stored by a `storage` technology in each timestep.
    default: 0
    unit: energy
    foreach:
    - nodes
    - techs
    - timesteps
    where: include_storage=True OR base_tech=storage
    bounds:
      min: 0
      max: .inf
  purchased_units:
    description: "Integer number of a technology that has been purchased,\nfor any
      technology set to require integer capacity purchasing.\nThis is used to allow
      installation of fixed capacity units of technologies (\nif `flow_cap_max` ==
      `flow_cap_min`) and/or to set a fixed cost for a technology,\nirrespective of
      its installed capacity.\nOn top of a fixed technology cost,\na continuous cost
      for the quantity of installed capacity can still be applied.\n\nSince technology
      capacity is no longer a continuous decision variable,\nit is possible for these
      technologies to have a lower bound set on outflow/consumption\nwhich will only
      be enforced in those timesteps that the technology is operating.\nOtherwise,
      the same lower bound forces the technology to produce/consume\nthat minimum
      amount of carrier in *every* timestep.\n"
    default: 0
    unit: integer
    foreach:
    - nodes
    - techs
    where: cap_method=integer
    domain: integer
    bounds:
      min: purchased_units_min
      max: purchased_units_max
  operating_units:
    description: Integer number of a technology that is operating in each timestep,
      for any technology set to require integer capacity purchasing.
    default: 0
    unit: integer
    foreach:
    - nodes
    - techs
    - timesteps
    where: integer_dispatch=True AND cap_method=integer
    domain: integer
    bounds:
      min: 0
      max: .inf
  available_flow_cap:
    description: Flow capacity that will be set to zero if the technology is not operating
      in a given timestep and will be set to the value of the decision variable `flow_cap`
      otherwise.
    default: 0
    unit: power
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: integer_dispatch=True AND flow_cap_max AND NOT flow_cap_per_unit
    bounds:
      min: 0
      max: .inf
  async_flow_switch:
    description: Binary switch to force asynchronous outflow/consumption of technologies
      with both `flow_in` and `flow_out` defined. This ensures that a technology with
      carrier flow efficiencies < 100% cannot produce and consume a flow simultaneously
      to remove unwanted carrier from the system.
    default: 0
    unit: integer
    foreach:
    - nodes
    - techs
    - timesteps
    where: force_async_flow=True
    domain: integer
    bounds:
      min: 0
      max: 1
  unmet_demand:
    description: Virtual source of carrier flow to ensure model feasibility. This
      should only be considered a debugging rather than a modelling tool as it may
      distort the model in other ways due to the large impact it has on the objective
      function value. When present in a model in which it has been requested, it indicates
      an inability for technologies in the model to reach a sufficient combined supply
      capacity to meet demand.
    default: 0
    unit: energy
    foreach:
    - nodes
    - carriers
    - timesteps
    where: config.ensure_feasibility=True
    bounds:
      min: 0
      max: .inf
  unused_supply:
    description: 'Virtual sink of carrier flow to ensure model feasibility. This should
      only be considered a debugging rather than a modelling tool as it may distort
      the model in other ways due to the large impact it has on the objective function
      value. In model results, the negation of this variable is combined with `unmet_demand`
      and presented as only one variable: `unmet_demand`. When present in a model
      in which it has been requested, it indicates an inability for technologies in
      the model to reach a sufficient combined consumption capacity to meet required
      outflow (e.g. from renewables without the possibility of curtailment).'
    default: 0
    unit: energy
    foreach:
    - nodes
    - carriers
    - timesteps
    where: config.ensure_feasibility=True
    bounds:
      min: -.inf
      max: 0
objectives:
  min_cost_optimisation:
    description: Minimise the total cost of installing and operating all technologies
      in the system. If multiple cost classes are present (e.g., monetary and co2
      emissions), the weighted sum of total costs is minimised. Cost class weights
      can be defined in the indexed parameter `objective_cost_weights`.
    equations:
    - where: any(cost, over=[nodes, techs, costs])
      expression: "sum(\n  sum(cost, over=[nodes, techs])\n  * objective_cost_weights,\n\
        \  over=costs\n) + $unmet_demand"
    - where: NOT any(cost, over=[nodes, techs, costs])
      expression: $unmet_demand
    sub_expressions:
      unmet_demand:
      - where: config.ensure_feasibility=True
        expression: "sum(\n  sum(unmet_demand - unused_supply, over=[carriers, nodes])\n\
          \  * timestep_weights,\n  over=timesteps\n) * bigM"
      - where: NOT config.ensure_feasibility=True
        expression: '0'
    sense: minimise
    active: true
global_expressions:
  flow_out_inc_eff:
    description: Outflows after taking efficiency losses into account.
    default: 0
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_out
    equations:
    - where: base_tech=transmission
      expression: "flow_out / (\n  flow_out_eff * flow_out_parasitic_eff *\n  flow_out_eff_per_distance
        ** distance\n)"
    - where: NOT base_tech=transmission
      expression: flow_out / (flow_out_eff * flow_out_parasitic_eff)
  flow_in_inc_eff:
    description: Inflows after taking efficiency losses into account.
    default: 0
    foreach:
    - nodes
    - techs
    - carriers
    - timesteps
    where: flow_in
    equations:
    - where: base_tech=transmission
      expression: flow_in * flow_in_eff * flow_in_eff_per_distance ** distance
    - where: NOT base_tech=transmission
      expression: flow_in * flow_in_eff
  cost_var:
    description: The operating costs per timestep of a technology.
    default: 0
    unit: cost_per_time
    foreach:
    - nodes
    - techs
    - costs
    - timesteps
    where: cost_export OR cost_flow_in OR cost_flow_out
    equations:
    - expression: timestep_weights * ($cost_export + $cost_flow_out + $cost_flow_in)
    sub_expressions:
      cost_export:
      - where: flow_export
        expression: sum(cost_export * flow_export, over=carriers)
      - where: NOT flow_export
        expression: '0'
      cost_flow_in:
      - where: base_tech=supply
        expression: cost_flow_in * source_use
      - where: NOT base_tech=supply
        expression: sum(cost_flow_in * flow_in, over=carriers)
      cost_flow_out:
      - expression: sum(cost_flow_out * flow_out, over=carriers)
  cost_investment_flow_cap:
    description: The investment costs associated with the nominal/rated capacity of
      a technology.
    default: 0
    foreach:
    - nodes
    - techs
    - carriers
    - costs
    where: flow_cap AND (cost_flow_cap OR cost_flow_cap_per_distance)
    equations:
    - expression: $cost_sum * flow_cap
    sub_expressions:
      cost_sum:
      - where: base_tech=transmission
        expression: (cost_flow_cap + cost_flow_cap_per_distance * distance) * 0.5
      - where: NOT base_tech=transmission
        expression: cost_flow_cap
  cost_investment_storage_cap:
    description: The investment costs associated with the storage capacity of a technology.
    default: 0
    foreach:
    - nodes
    - techs
    - costs
    where: cost_storage_cap AND storage_cap
    equations:
    - expression: cost_storage_cap * storage_cap
  cost_investment_source_cap:
    description: The investment costs associated with the source consumption capacity
      of a technology.
    default: 0
    foreach:
    - nodes
    - techs
    - costs
    where: cost_source_cap AND source_cap
    equations:
    - expression: cost_source_cap * source_cap
  cost_investment_area_use:
    description: The investment costs associated with the area used by a technology.
    default: 0
    foreach:
    - nodes
    - techs
    - costs
    where: cost_area_use AND area_use
    equations:
    - expression: cost_area_use * area_use
  cost_investment_purchase:
    description: The investment costs associated with the binary purchase of a technology.
    default: 0
    foreach:
    - nodes
    - techs
    - costs
    where: cost_purchase AND purchased_units
    equations:
    - where: base_tech=transmission
      expression: (cost_purchase + cost_purchase_per_distance * distance) * purchased_units
        * 0.5
    - where: NOT base_tech=transmission
      expression: cost_purchase * purchased_units
  cost_investment:
    description: The installation costs of a technology, including annualised investment
      costs and annual maintenance costs.
    default: 0
    unit: cost
    foreach:
    - nodes
    - techs
    - costs
    where: cost_investment_flow_cap OR cost_investment_storage_cap OR cost_investment_source_cap
      OR cost_investment_area_use OR cost_investment_purchase
    equations:
    - expression: "$annualisation_weight * (\n  $depreciation_rate * (\n    sum(default_if_empty(cost_investment_flow_cap,
        0), over=carriers) +\n    default_if_empty(cost_investment_storage_cap, 0)
        +\n    default_if_empty(cost_investment_source_cap, 0) +\n    default_if_empty(cost_investment_area_use,
        0) +\n    default_if_empty(cost_investment_purchase, 0)\n  ) * (1 + cost_om_annual_investment_fraction)\n\
        \  + sum(cost_om_annual * flow_cap, over=carriers)\n)\n"
    sub_expressions:
      annualisation_weight:
      - expression: sum(timestep_resolution * timestep_weights, over=timesteps) /
          8760
      depreciation_rate:
      - where: cost_depreciation_rate
        expression: cost_depreciation_rate
      - where: NOT cost_depreciation_rate AND cost_interest_rate=0
        expression: 1 / lifetime
      - where: NOT cost_depreciation_rate AND cost_interest_rate>0
        expression: (cost_interest_rate * ((1 + cost_interest_rate) ** lifetime))
          / (((1 + cost_interest_rate) ** lifetime) - 1)
  cost:
    description: The total annualised costs of a technology, including installation
      and operation costs.
    default: 0
    unit: cost
    foreach:
    - nodes
    - techs
    - costs
    where: cost_investment OR cost_var
    equations:
    - expression: $cost_investment + $cost_var_sum
    sub_expressions:
      cost_investment:
      - where: cost_investment
        expression: cost_investment
      - where: NOT cost_investment
        expression: '0'
      cost_var_sum:
      - where: cost_var
        expression: sum(cost_var, over=timesteps)
      - where: NOT cost_var
        expression: '0'
    active: true
BTLF >      +�             �*�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OCHK    ��                      NAME          nodes +        _Netcdf4Dimid                   	 �{     �       �                                                                                                                                                                          �'7�       ��OHDR(                                     *       i�      h       ��     �           " ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          techs   �xV�OHDR+                                     *       i�      �       �x     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE #        NAME    	      carriers   +F��OHDRM    �      �                @    *         �    L�     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE $        NAME    
      timesteps +        _Netcdf4Dimid                  6�-C    ?:�yFRHP               ��������      ��                                                                                  (  �        �4�BTHD       d(��              j�\BTHD 	      d(��              �\:&                  GCOL                        math                  config                defaults
              ��      h               l              ��      �               �               �               �               �              aluminium_transmission_line     �              fuel_supply     �              aluminium_supply�              aluminium_demand�               �              m�     �              ��      �               �               �              Netherlands     �              Iceland �               �              m�     �              ��      �              �      �               �               �              fuel    �       	       aluminium                    m�                  ��                   �                   �                   m�                  ��                   �                   �                   m�                  ��                   �                   m�                  �                   �                   m�                  ��                   �                   �                    m�     !             ��      "             �      #             �      -              .             m�     /             ��      0             i�      1             �      2             m�     3             ��      4             i�      5              6             monetary=             m�     >             ��      ?             �      @             �      B             ��      C             �      G             �      H             ��      I             i�      K             �      L             i�      M             i�      N              O             ��      P              Q              R              S              T             transmission    U             supply  V             supply  W             demand  X              Z             m�     [             ��      \              ]              ^              _              `              a              b              c              d      	       aluminium       e              f              g              h              i             fuel    j      	       aluminium       k              o             m�     p             ��      q             �      r              s             ��      t              u              v              w              x             #8465A9 y             #E37A72 z             #F9CF22 {             #072486 }             ��      ~             i�      �             ��      �             i�      �             ��      �              �             ��      �              �              �              �              �             aluminium Transmission Line     �             Fuel Supply     �             Aluminium Supply�             Aluminium Demand�              �             ��      �              �              �              �              �              �             cost_dim_setter �             cost_dim_setter �              �             m�     �             ��      �             �      �             ��      �             ��      �             ��      �             m�     �             m�     �             �      �             m�     �             ��      �             �      �             m�     �             ��      �             m�     �             ��      �             �      �             ��      �             �      �             �              �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDR(                                     *       i�      -      ��     Q            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          costs   - �          �a��OHDR�4                                                  ?      @ 4 4�     +         �                   x�                       y�      �      1�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               �c��FSHD                               P x (        �                    *� �BTLF  _    D     �$ 3   1  	   +�
3         �I75 f   7     ��9 �   +     e>�    2     Ŵ�� �   2     �]Y� �   2     f�� 1   5     bS�� d   l  
   c���     I      c��� �    o     G���nF��                                                                                                                                                                                                                                                                                                          BTLF 	     I       _    D      �    o               1   5      f   7      �   2      �   2         2      3   1  	    d   l  
    �   +     ����                                                                                                                                                                                                                                                                                                                                                          FSSE R     * �    GZ%kFSSE Dx     �      �B�FSSE      
  �   1��XBTHD       d(�M             n��_BTHD       d(8�     	 	       ��'�   K �    ��hBTHD       d(��              ܀}BTHD 	      d(��               ��FSSE ��       �    �4nFRHP               ��������                                +                                                         ��      H���           ���FSSE �
        �    l  �/    l �&    �              ��� FSSE a�        �    ��uGFHDB y�           n��     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     description    N      A technology's flow capacity, also known as its nominal or nameplate capacity.     unit          power     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �      i�      �      i�      �        _Netcdf4Dimid                                                                                                                                                                                                                                                                                      FHDB �	       (   �4�     _model_def_dict    �     nodes:
  Iceland:
    latitude: 64.1355
    longitude: -21.8954
    techs:
      aluminium_supply:
        carrier: aluminium
      fuel_supply:
        carrier: fuel
  Netherlands:
    latitude: 52.3676
    longitude: 4.9041
    techs:
      aluminium_demand:
        carrier: aluminium
parameters:
  bigM: 1000000.0
  objective_cost_weights:
    data: 1
    dims: costs
    index:
    - monetary
tech_groups:
  cost_dim_setter:
    cost_area_use:
      data:
      dims: costs
      index: monetary
    cost_flow_cap:
      data:
      dims: costs
      index: monetary
    cost_flow_in:
      data:
      dims: costs
      index: monetary
    cost_flow_out:
      data:
      dims: costs
      index: monetary
    cost_interest_rate:
      data: 0.1
      dims: costs
      index: monetary
    cost_source_cap:
      data:
      dims: costs
      index: monetary
    cost_storage_cap:
      data:
      dims: costs
      index: monetary
techs:
  aluminium_demand:
    base_tech: demand
    carrier_in: aluminium
    color: '#072486'
    name: Aluminium Demand
  aluminium_supply:
    base_tech: supply
    carrier_out: aluminium
    color: '#F9CF22'
    cost_flow_out:
      data: 0.002
    flow_cap_max: 10000
    inherit: cost_dim_setter
    name: Aluminium Supply
  aluminium_transmission_line:
    base_tech: transmission
    carrier_in: aluminium
    carrier_out: aluminium
    color: '#8465A9'
    flow_cap_max: 10000
    flow_out_eff: 1
    from: Iceland
    fuel_consumption_rate: 0.05
    lifetime: 20
    name: aluminium Transmission Line
    to: Netherlands
  fuel_supply:
    base_tech: supply
    carrier_out: fuel
    color: '#E37A72'
    cost_flow_out:
      data: 0.02
    flow_cap_max: 40000
    inherit: cost_dim_setter
    name: Fuel Supply
     serialised_sets  ?      @ 4 4�         _NCProperties    "      version=2,netcdf=4.9.2,hdf5=1.14.3                                                                                                            FHDB �	           s�'� 	    defaults          available_area: .inf
bigM: 1000000000.0
objective_cost_weights: 1
area_use: .inf
area_use_max: .inf
area_use_min: 0
area_use_per_flow_cap: .nan
cap_method: continuous
color: .nan
cost_area_use: 0
cost_depreciation_rate: 1
cost_export: 0
cost_flow_cap: 0
cost_flow_cap_per_distance: 0
cost_flow_in: 0
cost_flow_out: 0
cost_interest_rate: 0
cost_om_annual: 0
cost_om_annual_investment_fraction: 0
cost_purchase: 0
cost_purchase_per_distance: 0
cost_source_cap: 0
cost_storage_cap: 0
cyclic_storage: true
distance: 1.0
export_max: .inf
flow_cap: .inf
flow_cap_max: .inf
flow_cap_max_systemwide: .inf
flow_cap_min: 0
flow_cap_min_systemwide: 0
flow_cap_per_storage_cap_max: .inf
flow_cap_per_storage_cap_min: 0
flow_cap_per_unit: .nan
flow_in_eff: 1.0
flow_in_eff_per_distance: 1.0
flow_out_eff: 1.0
flow_out_eff_per_distance: 1.0
flow_out_min_relative: 0
flow_out_parasitic_eff: 1.0
flow_ramping: 1.0
force_async_flow: false
include_storage: false
integer_dispatch: false
lifetime: .inf
name: .nan
one_way: false
purchased_units: .inf
purchased_units_max: .inf
purchased_units_max_systemwide: .inf
purchased_units_min: 0
purchased_units_min_systemwide: 0
sink_unit: absolute
sink_use_equals: .nan
sink_use_max: .inf
sink_use_min: 0
source_cap: .inf
source_cap_equals_flow_cap: false
source_cap_max: .inf
source_cap_min: 0
source_eff: 1.0
source_unit: absolute
source_use_equals: .nan
source_use_max: .inf
source_use_min: 0
storage_cap: .inf
storage_cap_max: .inf
storage_cap_min: 0
storage_cap_per_unit: .nan
storage_discharge_depth: 0
storage_initial: 0
storage_loss: 0
     allow_operate_mode                                                                                                                                                                                                                                                                                                                                                                                                       FHDB �	           ��W�     termination_condition          optimal     calliope_version_defined          0.7.0     calliope_version_initialised    
      0.7.0.dev3     applied_overrides            	    scenario          None     config    �     build:
  backend: pyomo
  ensure_feasibility: true
  mode: plan
  objective: min_cost_optimisation
  operate_use_cap_results: false
  add_math:
  - custom_constraints_delay.yaml
  - custom_constraints_state.yaml
solve:
  save_logs:
  solver: glpk
  solver_io:
  solver_options:
  spores_number: 3
  spores_save_per_spore: false
  spores_score_cost_class: spores_score
  spores_skip_cost_op: false
  zero_threshold: 1e-10
     applied_additional_math  ?      @ 4 4�         name          2-node model     serialised_dicts                               i�            i�            i�              serialised_bools  ?      @ 4 4�         serialised_nones          scenario             FHIB �	           �      ���������������������������������������������������������      �      ����������������!F�ETREE  ����������������1                                       U�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�                      ?      @ 4 4�     +         �                   ��                 ��      �      ;�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    �ݥ�FRHP               ��������       a�                                                                                  (  ��        	�PBTHD       d(��              "�L�BTHD 	      d(��              6{BqFSHD                                P x (        �                    ��BBTLF  W    D     �$ 8   1  	   +�
3         �I75 k   7     ��9 i   +     e>�    2     Ŵ�� �   2     �]Y� �   2     f�� 6   5     bS�� �   L     c���     A      c��� �    |     G�����d�                                                                                                                                                                                                                                                                                                          BTLF 	     A       W    D      �    |               6   5      k   7      �   2      �   2         2      8   1  	    i   +      �   L     �d�+                                                                                                                                                                                                                                                                                                                                                          FRHP               ���������       ��                                                                                  (  	S       �o!�           ��Gx^c`����:�����b`���0�`��b��Q_� ���� V}= ��0FHDB ��           -���     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      �     description    [      A transmission technology's flow capacity, also known as its nominal or nameplate capacity.     unit          power     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         _Netcdf4Dimid                     DIMENSION_LIST                              i�      l                                                                                                                                                                                                                                                                                                   TREE  ����������������                       Q�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OHDRWD                         �                           �          ?      @ 4 4�     +         �                   g              �          �      ��      ��       v                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         .d(FRHP               ���������       ��                                                                                  (  g�        9�y�FSHD  �                              P x (        Y                   UJTcBTLF  c    D     �$ �   1  	   +�
3 r         �I75 �   7     ��9 A   +     e>� b   2     Ŵ�� 0   2     �]Y� �   2     f�� �   5     bS�� �   |  
   c���     M      c��� �    �     G����~                                                                                                                                                                                                                                                                                                          BTLF 	     M       c    D      �    �      r          �   5      �   7      �   2      0   2      b   2      �   1  	    �   |  
    A   +     ��'�                                                                                                                                                                                                                                                                                                                                                          FRHP               ��������      �                                                                                 (  /�       7�9'BTHD 	      d(X�             
u��FSSE      ,  �   B��BTHD 	      d(�O             ��KFSHD  �                              P x (        UR                   !{q�      ��Xx^c`�:~�8���ޡ !��FHDB �           �x��     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     description    �      The outflow of a technology per timestep, also known as the flow discharged (from `storage` technologies) or the flow received (by `transmission` technologies) on a link.     unit          energy     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�           i�           i�           i�             _Netcdf4Dimid                                                                                                                                                                     TREE  �����������������                                                                                 �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR D                         �                           �          ?      @ 4 4�     +         �                   �$              �          �     f     �                                         �J��BTHD       d(Y     + +       �%
BTHD      d(&      +       >Y�8FSHD  �                              P x          �     ,       ,       ]�DBTLF �?� >   i�8 �   y}W !   ���  V  # �rQ' �    5�. �   �8 �    ���I �  ) )m�M 0  & y��P |    `��P    ��-S B  , 6�gV \   w�iV �   ���X �   ���[ ]  # �^P^ w   ��l �     'q �  " ��{� )   q��� �   �� 4  ( 1M7� �  " D9�    �9p� �   �ԕ�    ��_� `    J鱷 n  ' 7���   - ĕ�� �    .BD� A    v��� a  % ho�     ��� �   S�:�    Ѧ� -    d�� E    �8�� �  $ �;�� �  " yܴ� �  ! �}"� �   XX� �  $ ��� �   )0                             BTLF         �            �             �            �    	           
        !           >           ]  #         �  "         �           �           -                        E            `            |    �V�F                                                                                                                                                                                                                                                                      FSSE ��      �  l    �7!�FRHP               ��������*      R                                                                                 (  ͔       ے0 Aa            �&FRHP               ��������)      74                                                                                 (  �y       ����           �O`(OHDR 4                  �                    �         +         �                   s>           �          �4     "5     H5         ?      @ 4 4�   3                                                       ӴA           FRHP               ���������       �                                                                                 (  �       i��BTHD       d(             a� BTHD 	      d(             �;FSHD  �                              P x (        )x                   IZN�BTLF  c    D     �$ �   1  	   +�
3 i         �I75 �   7     ��9 8   +     e>� Y   2     Ŵ�� '   2     �]Y� �   2     f�� �   5     bS�� �   |  
   c���     M      c��� �    �     G���4���                                                                                                                                                                                                                                                                                                          BTLF 	     M       c    D      �    �      i          �   5      �   7      �   2      '   2      Y   2      �   1  	    �   |  
    8   +     �Y.�                                                                                                                                                                                                                                                                                                                                                          x^�ڱ� CAF�fi�W�J�� �n��/$3                                                          � �     ��A�� `U ^8@v����{ ���e�  �<�A�� `U7 .8@v��4���S�wh��  ��n��{ �U��/�FHDB �          �     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     description    �      The inflow to a technology per timestep, also known as the flow consumed (by `storage` technologies) or the flow sent (by `transmission` technologies) on a link.     unit          energy     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�           i�           i�           i�             _Netcdf4Dimid                                                                                                                                                                              FHDB �          ��X%       nodesm�            techs��             carriers�             	timesteps�             costsi�             flow_cap��             link_flow_cap��             flow_out��             flow_in�     	       
source_use�     
       
source_cap�J            unmet_demandAa            flow_out_inc_eff�p            flow_in_inc_eff�            cost_var�            costw�            capacity_factor               TREE  �����������������                                               �9                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������       Dx                                                                                 (  ��       ��VoBTHD       d(m�             �[�BTHD 	      d(m�             ;�<B?YGBTHD       d(�s             `�vBTHD 	      d(�u             _�]�FSHD  )                             P x (        �                   ;�ϡ       FRHP               ���������       n5                                                                                 (  s:       �]BTHD       d(�5             �E�UBTHD 	      d(�7             ���FSHD  �                              P x (        ]�                   �ȩ=BTLF  _    D     �$ <   1  	   +�
3          �I75 o   7     ��9 �   +     e>� 
   2     Ŵ�� �   2     �]Y� �   2     f�� :   5     bS�� m   l  
   c���     I      c��� �    w     G�����                                                                                                                                                                                                                                                                                                          BTLF 	     I       _    D      �    w                :   5      o   7      �   2      �   2      
   2      <   1  	    m   l  
    �   +     .��                                                                                                                                                                                                                                                                                                                                                          x^�ڡ� A*��X0�����3؃��a�.                                                   ��d�       �s�0@v ���{    �;?  ��d�  �V � h���Qv �j  jl0@v ��d�  ��Ӆ7�� Z�YFHDB �4          sZ�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     description    V      The carrier flow consumed from outside the system boundaries by a `supply` technology.     unit          energy     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�           i�           i�             _Netcdf4Dimid                                                                                                                                                                                                                                                                             TREE  ����������������p                                       �R                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR�$                                    ?      @ 4 4�     +         �                   	W                   ��      =�      ��       ?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  �r�BTLF  [    D     �$ ]   1  	   +�
3 <        �I75 �   7     ��9 �   +     e>� +   2     Ŵ�� �   2     �]Y� �   2     f�� [   5     bS�� �   \  
   c���     E      c��� �    �     G���g��$                                                                                                                                                                                                                                                                                                          BTLF 	     E       [    D      �    �      <         [   5      �   7      �   2      �   2      +   2      ]   1  	    �   \  
    �   +      V��                                                                                                                                                                                                                                                                                                                                                          BTHD       d(�             �6�BTHD 	      d(�             y���FSHD  *                             P x (        ��                    haOFSSE ��      �      ��( BTHD       d(X�             xc3��>x^�ء�@C�+�ΰ�ESH|$���٭�L�                            ��.(��   ��n(���D�c`��  ����a��a��  ���6�FHDB ��           95�     _Netcdf4Coordinates                                    _FillValue  ?      @ 4 4�                      �     description    |      The upper limit on a flow that can be consumed from outside the system boundaries by a `supply` technology in each timestep.     unit          power     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �      i�      �        _Netcdf4Dimid                                                                                                                                                                                                                                                            TREE  ����������������                               ;d                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�4                  �                    �          ?      @ 4 4�     +         �                   Wd           �         	 �w     c�      7�      !                                                                            D                                                                       7                                                          2                                                     2                                                     2                                                     1                                                    l                                                                                                               "Xmx^c` t@��0� �z�z�z   kM
VTREE  ����������������L                                       {y                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR�D                         �                           �          ?      @ 4 4�     +         �                   �}              �               �3     4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ���BTLF  c    D     �$ �   1     +�
3 2   7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� i   2     f�� �    5     bS�� 0   |  	   c���     M      c��� �    V     G�����N�                                                                                                                                                                                                                                                                                                                           BTLF 	     M       c    D      �    V      �    5      2   7      i   2      �   2      �   2      �   1      0   |  	    �   +  
   �a�                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������      X�                           	                                                      (  x�       �Y�(FSSE �     �  c    �S|FSHD  �                              P x (        �                    �4OCHK    ��             +        _Netcdf4Dimid                   6     :       v                                                                                                                          D��*x^��1  AQ�c+ئ���Ѱ��            ��� ^5 @� �UP���&��y�2�FHDB           �U�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     description    5      Outflows after taking efficiency losses into account.     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�           i�           i�           i�             _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                          TREE  �����������������                                               �                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�D                         �                           �          ?      @ 4 4�     +         �                   ͘              �          t     �Q     �Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           X�ŋBTLF  c    D     �$ �   1     +�
3 1   7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� h   2     f�� �    5     bS�� /   |  	   c���     M      c��� �    U     G�����7�                                                                                                                                                                                                                                                                                                                           BTLF 	     M       c    D      �    U      �    5      1   7      h   2      �   2      �   2      �   1      /   |  	    �   +  
   "O                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������      ��                           
                                                      (  f�       <�Y_�6BFSHD  �                             P x (        &�                   ڸ�                  �8�x^�ڱ� CAF�fi�W�J�� �n��/$3                                                          � �     ��A�� `U ^8@v����{ ���e�  �<�A�� `U7 .8@v��4���S�wh��  ��n��{ �U��/�FHDB t          d�H�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     description    4      Inflows after taking efficiency losses into account.     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�            i�      !     i�      "     i�      #       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                           TREE  �����������������                                               |�                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�D                         �                           �          ?      @ 4 4�     +         �                   /�              �          ��      pR     ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ����FSHD                               P x (        B�                   �i�BTLF  c    D     �$ "   1  	   +�
3 �    '     �I75 U   7     ��9 �   +     e>� �   2     Ŵ�� �   2     �]Y� �   2     f��     5     bS�� S   |  
   c���     M      c��� �    R     G���LB8�                                                                                                                                                                                                                                                                                                          BTLF 	     M       c    D      �    R      �    '          5      U   7      �   2      �   2      �   2      "   1  	    S   |  
    �   +     (��<                                                                                                                                                                                                                                                                                                                                                          FSHD  �                             P x (        ]�                   �ەEOCHK    T�     @       +        _Netcdf4Dimid                   �     J       V                                                                                          � FRHP               ���������      ZL                           	                                                      (  �T       ��2�FSSE ZL     � 8    ʽ1̢FSSE 74     ) �    ��<dBTHD 	      d(8�     	 	       Y/H�FSSE n5     �      ��4���x^�ڡ� A*��X0�����3؃��a�.                                                   ��d�       �s�0@v ���{    �;?  ��d�  �V � h���Qv �j  jl0@v ��d�  ��Ӆ7�� Z�YFHDB ��           ���     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     description    1      The operating costs per timestep of a technology.     unit          cost_per_time     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      .     i�      /     i�      0     i�      1       _Netcdf4Dimid                                                                                                                                                                                                                                                                                       TREE  ����������������i                                               ��                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�4                                                  ?      @ 4 4�     +         �                   ��                      	3     �3     �3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ��BTLF  _    D     �$ ;   1  	   +�
3         �I75 n   7     ��9 �   +     e>� 	   2     Ŵ�� �   2     �]Y� �   2     f�� 9   5     bS�� l   l  
   c���     I      c��� �    x     G�����>                                                                                                                                                                                                                                                                                                          BTLF 	     I       _    D      �    x               9   5      n   7      �   2      �   2      	   2      ;   1  	    l   l  
    �   +     %d5                                                                                                                                                                                                                                                                                                                                                          OHDR                                     *       i�      �       i�      9           ������������������������A         _Netcdf4Coordinates                            +        CLASS          DIMENSION_SCALE   ����w�             ��Mx^��A  AD! g��ɿ ��Ѱ��c                                  �;��@v厁��{  ��,h ����g���{  ���%ݷFHDB 	3          ���     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     description    W      The total annualised costs of a technology, including installation and operation costs.     unit          cost     default                             
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      2     i�      3     i�      4       _Netcdf4Dimid                                                                                                                                                                                                                                                                              TREE  ����������������                                       8�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTLF  _    D     �$ p   1     +�
3 �    7     ��9    +     e>� >   2     Ŵ��    2     �]Y� �    2     f�� �   l     c���     I      c���0��T                                                                                                                                                                                                                                                                                                                                                             BTLF 	     I       _    D      �    7      �    2         2      >   2      p   1      �   l         +     _�	K                                                                                                                                                                                                                                                                                                                                                                                                 FRHP               ���������      ��                           	                                                      (  <       ��l�BTHD       d(�3     	 	       d���FSSE �j     � B    Vh@1   {���BTHD       d(�     	 	       �fnd      j)gFSSE �      �    x^FSSE X�     � 8    �x�iFHDB �w          ��D�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�           i�           i�             _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FRHP               ���������      0�                           	                                                      (  �%       ?��BTHD 	      d(�     	 	       7�'�FSHD  �                             P x (        ��                   �ՍFSSE 0�     � L    �^5;FHIB �          �"     �#     O     ��������Hs�BTHD 	      d(�5     	 	       "]p�FSHD  �                             P x (        J                   �p�8�OCHK   ��     B      :        units          hours since 2005-01-01 00:00:001    	    calendar          proleptic_gregorian  �;�bOCHK  	 ��     �       +        _Netcdf4Dimid                  wg3`    x^c` @�g0� �z�z�z   a	�   i�      �      i�      �      i�      �      i�      �      i�      �      i�      �   	   i�      �      i�      �                                                                   	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �         i�      6  x^��1 0�0+6����JQbc���                                                                      ��d�      ��&��  �d�      ��&�� �z
$�x^c` |`x �큐� VOx^c�� +��G(}	J�B�$�#���׃ )�x^�ð��#�%_�$���� 6�.x^c` �� '0    ��.A   i�      W     i�      V     i�      T     i�      U      i�      k  	   i�      j      i�      h     i�      i  	   i�      d      i�      e      i�      f      i�      g  x^c`�-��������B� ��Q   i�      {     i�      z     i�      x     i�      y  x^c��PͰ�A��#�:�K����~$��Ro_o �Wx^c��0�a&&�����z�z{ ��x^c`�:::~>����� )�9   i�      �     i�      �     i�      �     i�      �      i�      �     i�      �      i�      �     i�      �  x^c`�-��p=�#a ���x^c`�~���޾ ��x^c`����0����G}�}= c��x^c`�~�0�Q_�P 0�x^{i��h��Ѳ��C?��� (y`x^[�p�����d�3O�
p  G-x^�š   �N�,����]=2�"�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�����$I�$I�$���I�$I�$I�dq��F�$I�$I�$���7�$I�$I�$�<R���x^�ű  ��H#Q����p/���$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I��5�$I�$I�$�8H�$I�$I���ܕ3�$I�$I�$��]=#I�$I�$I�����@x^c``BF �  ? x^c`x���`���x!���?����w� k4	x^��1  �5�#X�o@   ���JOp�x^��1  �5�#X�o@   ���JOp�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                OHDRD                         �                           �          ?      @ 4 4�     +         �                   j              �         	 x�     �     
�      5                                                                                D                                                                       7                                                          2                                                     2                                                     2                                                     1                                                    |                                                                                                                               ��COCHK,    0   REFERENCE_LIST 6     dataset        dimension                         ��             ��             �            Aa            �p            �                        �            �7             �P             ��            ~�            �*            �q�g���FRHP               ���������      �b                           	                                                      (  �r       i���BTHD       d(8c     	 	       >��9FSSE �b     � $    ��2��@FSSE ��     � $    ���� ,"�hTREE  �����������������                                               $                                   �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF  c    D     �$ t   1     +�
3 �    7     ��9 !   +     e>� B   2     Ŵ��    2     �]Y� �    2     f�� �   |     c���     M      c����K�                                                                                                                                                                                                                                                                                                                                                             BTLF 	     M       c    D      �    7      �    2         2      B   2      t   1      �   |      !   +     Xn�                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                    ?      @ 4 4�     +         �                   �)                  	 8�     ��     ��                                                                              D                                                                       7                                                          2                                                     2                                                     2                                                     1                                                    \                                                                                               ��A�FRHP               ���������      J�                           
                                                      (  ��        &4/BTHD       d(��     
 
       ����FSSE �;     � V    ��d��BTHD       d(�L     	 	       �úBTHD 	      d(�N     	 	       _h��     {;�$FHDB �         �]       systemwide_capacity_factor�            systemwide_levelised_cost�7            total_levelised_cost�P            objective_cost_weights8g            bigM(            	base_techP�            carrier�            carrier_out��            color��            cost_flow_out�             cost_interest_rateT�     !       flow_cap_max��     "       name��     #       techs_inheritanceD�     $       
carrier_in~�        FHDB x�          ��rI     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      =     i�      >     i�      ?     i�      @       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                     TREE  ����������������                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  [    D     �$ l   1     +�
3 �    7     ��9 �   +     e>� :   2     Ŵ��    2     �]Y� �    2     f�� �   \     c���     E      c���)�˫                                                                                                                                                                                                                                                                                                                                                             BTLF 	     E       [    D      �    7      �    2         2      :   2      l   1      �   \      �   +     A!"�                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�4                                                  ?      @ 4 4�     +         �                   @                     	 m�     Y#     #      !                                                                            D                                                                       7                                                          2                                                     2                                                     2                                                     1                                                    l                                                                                                               �pOOCHK         @       +        _Netcdf4Dimid                ^6E(FSSE �     � 5    e�,�   �l��OCHK    Z     �       +        _Netcdf4Dimid                 ���FSHD  �                             P x (        <#                   �iGFSSE s�     � N    ����    x"��FHDB 8�          ����     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      B     i�      C       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������#                                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          FSHD  �                             P x (        ��                   4-1BTLF  _    D     �$ p   1     +�
3 �    7     ��9    +     e>� >   2     Ŵ��    2     �]Y� �    2     f�� �   l     c���     I      c���0��T                                                                                                                                                                                                                                                                                                                                                             BTLF 	     I       _    D      �    7      �    2         2      >   2      p   1      �   l         +     _�	K                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                    ?      @ 4 4�     +         �                   �X                  	 s          �b                                                                              D                                                                       7                                                          2                                                     2                                                     2                                                     1                                                    \                                                                                               ��,�FRHP               ���������      T                           	                                                      (  ��       \h|:FSHD  �                             P x (        ��                   �7�DBTHD       d(�     	 	       ��f�FHDB m�          ����     _Netcdf4Coordinates                                      _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      G     i�      H     i�      I       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        TREE  ����������������                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTHD 	      d(8e     	 	       "\m�FSHD  �                             P x (        +                   ��RBTLF  [    D     �$ l   1     +�
3 �    7     ��9 �   +     e>� :   2     Ŵ��    2     �]Y� �    2     f�� �   \     c���     E      c���)�˫                                                                                                                                                                                                                                                                                                                                                             BTLF 	     E       [    D      �    7      �    2         2      :   2      l   1      �   \      �   +     A!"�                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�                      ?      @ 4 4�     +         �                   �v               
 �i     Zj     �j      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  A�9~FRHP               ���������      �j                           
                                                      (  �n       ��oBTHD       d(�j     
 
       c�}BTHD 	      d(�l     
 
       �T+�FSHD  �                             P x (        ��                   y�h9BTLF  W    D     �$ �   1     +�
3 �    7     ��9    +  	   e>� h   2     Ŵ�� 9   /     �]Y�    2     f�� �    5     bS�� �   L     c���     A      c���R0�`                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    5         2      9   /      h   2      �   1      �   L         +  	   �2#                                                                                                                                                                                                                                                                                                                                                                                    FHDB �i          |��     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      M       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                              FHDB s          ����     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      K     i�      L       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  OHDR           ?      @ 4 4�     *         �                     ������������������������D         _FillValue  ?      @ 4 4�                      �7    
    is_result                            A        default  ?      @ 4 4�                    e��A2        serialised_dicts  ?      @ 4 4�    /        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    O��OHDR<                                     *       i�      N      �:     Q            ������������������������A         _Netcdf4Coordinates                           7    
    is_result                            2        serialised_dicts  ?      @ 4 4�    /        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    L        DIMENSION_LIST                              i�      O    y[�uOHDR4                                                 +         �                   ��                     	 ~S     ��     h�         ?      @ 4 4�   I         @ 4 4�                                                         6*�BTHD       d(��     	 	       �޼Q�OHDR`$                                                   *       i�      X      T;     Q            ������������������������E         _Netcdf4Coordinates                               7    
    is_result                            2        serialised_dicts  ?      @ 4 4�    /        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    \        DIMENSION_LIST                              i�      Z     i�      [    p���FRHP               ���������      �                           	                                                      (  O       �#BTHD 	      d(�     	 	       �Z9�FSHD  �                             P x (        1;                   �f           ��mBTHD 	      d(��     	 	       f.rOHDR�                                     *       i�      r      	 �     bT     r�         �     @        �    ��������@                                                      7                                                          A                                                                    2                                                     l7$rBTHD       d(j�     
 
       T���BTHD 	      d(j�     
 
       �a\YFSHD  �                             P x (        �;                   �&�FSSE J�     � V    '���BTHD       d(�     	 	       7ԣ���FSSE ��     � 
    ���BTHD 	      d(��     
 
       ���pFSHD  �                             P x (        Ŋ                   � ,� �e�FSSE T     � 5    ڧ�VrTREE  ����������������                                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTLF  _    D     �$ m   1     +�
3 �    7     ��9 
   +     e>� ;   2     Ŵ��    /     �]Y� �    2     f�� �   l     c���     I      c���F��                                                                                                                                                                                                                                                                                                                                                             BTLF 	     I       _    D      �    7      �    2         /      ;   2      m   1      �   l      
   +     NP�o                                                                                                                                                                                                                                                                                                                                                                                                 FRHP               ���������      �;                           
                                                      (  �       =��.BTHD       d(��     
 
       �b'BTHD 	      d(��     
 
       ��S�<"FHDB ~S          v��     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      o     i�      p     i�      q       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            FRHP               ���������      ��                           	                                                      (  �       ��BTHD 	      d(�     	 	       �C#�FSHD  �                             P x (        	�                   �[bBTLF  b   1     +�
3 W    7     ��9 �   +     e>� 0   2     Ŵ��    /     �]Y� �    2     f�� �    A     bS�� �   L     c���     A      c���H �                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    7      �    A      �    2         /      0   2      b   1      �   L      �   +     1��                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                    ?      @ 4 4�     +         �                   �                  
 ��     ��     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ��!BTLF  [    D     �$ �   1     +�
3 �    7     ��9 +   +  	   e>� l   2     Ŵ�� =   /     �]Y�    2     f�� �    5     bS�� �   \     c���     E      c���.p                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    5         2      =   /      l   2      �   1      �   \      +   +  	   jqc�                                                                                                                                                                                                                                                                                                                                                                                    OCHK                 �  
   0   REFERENCE_LIST 6     dataset        dimension                         �            w�            �7            �P            8g             �            T�            i�>9FRHP               ���������      s�                           
                                                      (  �       ��ۙ  �B�kFHDB ��          ��%9     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      }     i�      ~       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                          FHDB �          �O	     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      s       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      TREE  ����������������)                               5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�$                                    ?      @ 4 4�     +         �                   ��                  
 �"     #     $�      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ��BTLF  [    D     �$ �   1     +�
3 �    7     ��9 +   +  	   e>� l   2     Ŵ�� =   /     �]Y�    2     f�� �    5     bS�� �   \     c���     E      c���.p                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    5         2      =   /      l   2      �   1      �   \      +   +  	   jqc�                                                                                                                                                                                                                                                                                                                                                                                    OHDR�                                     *       i�      �      	 j�     ��     ��         �     @        �    ��������@                                                        7                                                          A                                                                    2                                                     ��FHDB �"          � �w     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �     i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������                               ^                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�                      ?      @ 4 4�     +         �                   �               
 ��     '�     M�      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              �<��BTLF  W    D     �$ �   1     +�
3 �    7     ��9 #   +  	   e>� t   2     Ŵ�� E   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c���n��I                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    A         2      E   /      t   2      �   1      �   L      #   +  	   z.��                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������      "�                           	                                                      (  ��       t BTHD 	      d(�     	 	       ��T_FSHD  �                             P x (        ��                   j�R�OCHK7    
    is_result                            2        serialised_dicts  ?      @ 4 4�    ��XFSSE "�     � 
    F�Y�    ��&�FHDB ��          �k     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       v                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR.                                     *       i�      �      �     @            ������������������������A         _Netcdf4Coordinates                               t�     }           �     �       +        _Netcdf4Dimid                q�0�OHDR`4                                                 +         �                   ��                     	 N�     ��     ��         ?      @ 4 4�   �         @ 4 4�                                                           D                                                                       seBTHD       d(�	     	 	       �o�-؁�BTLF  b   1     +�
3 W    7     ��9 �   +     e>� 0   2     Ŵ��    /     �]Y� �    2     f�� �    A     bS�� �   L     c���     A      c���H �                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    7      �    A      �    2         /      0   2      b   1      �   L      �   +     1��                                                                                                                                                                                                                                                                                                                                                                                                 OCHK/        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    L        DIMENSION_LIST                              i�      �  �_�nBTHD       d(�w     
 
       Z�  �QFSSE      �     S���FSSE *     � N    �BTHD       d(\     	 	       :��
BTHD 	      d(^     	 	       ��q�FSSE �     �     v��tBTHD       d(Zp     	 	       <��9P�FSSE      � N    �o-�  FRHP               ���������                                 
                                                      (  O       �v �BTHD       d(e     
 
       �n9���FHDB j�          2���     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDR�                      ?      @ 4 4�     +         �                   
 !�     ��     �         O            �                                                                                                                                                                                                                                                              ��5�FRHP               ���������      �                           	                                                      (  *d       U��FSHD  �                             P x (        ��                   ����BTHD 	      d(Zr     	 	       @~ǘBTHD 	      d(�y     
 
       �m�FSSE !S     �     �" OHDR�                      ?      @ 4 4�     +         �                   	 e     ��     �         �,                ��������                                                     D                                                                       7                                                          2                                                     �dBTHD       d(5     	 	       yݤr�KmTREE  ����������������                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                BTLF  _    D     �$ m   1     +�
3 �    7     ��9 
   +     e>� ;   2     Ŵ��    /     �]Y� �    2     f�� �   l     c���     I      c���F��                                                                                                                                                                                                                                                                                                                                                             BTLF 	     I       _    D      �    7      �    2         /      ;   2      m   1      �   l      
   +     NP�o                                                                                                                                                                                                                                                                                                                                                                                                 BTHD 	      d(e     
 
       � b�FSHD  �                             P x (        �                   ��wBTLF  W    D     �$ �   1     +�
3 �    7     ��9 #   +  	   e>� t   2     Ŵ�� E   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c���n��I                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    A         2      E   /      t   2      �   1      �   L      #   +  	   z.��                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������                                 	                                                      (  �C       �@�BTHD 	      d(7     	 	       ��v6FSHD  �                             P x (        >�                   ��R�FRHP               ���������      *                           
                                                      (  �?       ��.BTHD       d(�;     
 
       C��EBTHD 	      d(�=     
 
       .0#�tXFHDB !�          rw�7     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                  FHDB �         ^�y�%       flow_out_eff��     &       fuel_consumption_rate��     '       lifetime9     (       latitude�O     )       	longitude`     *       source_use_maxZt     +       sink_use_equals�b     ,       definition_matrix�*     -       distance��     .       timestep_resolution2�     /       timestep_weights,�                                                                                                                                            FHDB N�          ���     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �     i�      �     i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                       #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTIN         �  " Y     (     J�w                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF           -         B  ,         n  '         �  )         �           �           �                      )           A             a  % !        �   "        �   #        �  $ $        �   %           &        4  ( '        \   (        w   )        �   *        �  ! +        �  " ,        �  $ -           .        0  & /        V  # ����                                                                                                                FSHD  �                             P x (        Y�                   �J eOHDR4                                                            +   �    @�                     	 R}     �}     
~                     P                                                                    7                                                          2                                                     /                                                  2                                                                                       ��>�TREE  ����������������                       7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  W    D     �$ e   1     +�
3 �    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   L     c���     A      c���ȿZ]                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    D      �    7      �    2         /      3   2      e   1      �   L      �   +     W��                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�                      ?      @ 4 4�     +         �                   �G               
 o          '      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ����BTLF  W    D     �$ �   1     +�
3 �    7     ��9 #   +  	   e>� t   2     Ŵ�� E   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c���n��I                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    A         2      E   /      t   2      �   1      �   L      #   +  	   z.��                                                                                                                                                                                                                                                                                                                                                                                    FHDB o          �'P     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                  FHDB e          -ǟ     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������                       P                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                      ?      @ 4 4�     +         �                   �S               	 ��     t�     ��      �                                                                    D                                                                       7                                                          2                                                     /                                                  2                                                     1                                                    L                                                                               i�f�FRHP               ���������      !S                           	                                                      (  D�       D�c�FSHD  �                             P x (        ��                   .�wFSHD  �                             P x (        �S                   յLJFSSE sS     � v    h��j     >~�TREE  ����������������                       d                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  W    D     �$ e   1     +�
3 �    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   L     c���     A      c���ȿZ]                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    D      �    7      �    2         /      3   2      e   1      �   L      �   +     W��                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�                      ?      @ 4 4�     +         �                   *h               	 �R     ��     k�      �                                                                    D                                                                       7                                                          2                                                     /                                                  2                                                     1                                                    L                                                                               ][�;OHDRO4    �                    �                       +         �                   Ɵ     �               
 J�     ܓ     �         ?      @ 4 4�   �                                                                                                                                      ��:�           �Em�FHDB ��          �vr     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  W    D     �$ e   1     +�
3 �    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   L     c���     A      c���ȿZ]                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    D      �    7      �    2         /      3   2      e   1      �   L      �   +     W��                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�4    �                    �                        ?      @ 4 4�     +         �                   D�     �               
 &w     �     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ޓS�FRHP               ���������      sS                           
                                                      (  D~       2���BTLF  _    D     �$ �   1     +�
3 �    7     ��9 K   +  	   e>� |   2     Ŵ�� M   /     �]Y�    2     f�� �    A     bS�� �   l     c���     I      c���hk                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    A         2      M   /      |   2      �   1      �   l      K   +  	   ��"q                                                                                                                                                                                                                                                                                                                                                                                    OCHK�    0   REFERENCE_LIST 6     dataset        dimension                         ��              ��              �             �             �J             Aa             �p             �             �             w�                          �             ��             ~�             �O             `             Zt            �b            �*             .j!yFRHP               ���������      ��                           	                                                      (  f�       zS�VBTHD       d(��     	 	       ���BTHD 	      d(��     	 	       G�,�Zt            (�l�FHDB &w          ��0�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �     i�      �     i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                          FHDB �R          �%�"     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������r                                       �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTHD 	      d(�     
 
       �n��FSHD  �                             P x (        �                   ��|  �     ��aFSSE (�     � v    vx�FSSE ��     �     #5�         FRHP               ���������      (�                           
                                                      (  ƛ       t���BTHD       d(z�     
 
       lFmrBTHD 	      d(z�     
 
       ���FSHD  �                             P x (        �                   �\`KBTLF  _    D     �$ �   1     +�
3 �    7     ��9 K   +  	   e>� |   2     Ŵ�� M   /     �]Y�    2     f�� �    A     bS�� �   l     c���     I      c���hk                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    A         2      M   /      |   2      �   1      �   l      K   +  	   ��"q                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������      ��                           	                                                      (  ��       ����BTHD       d(,�     	 	       46�(BTHD 	      d(,�     	 	       ���jFSSE ��     �     \-o�BTHD       d(U�     	 	       ֛�5GzBTLF  Z        I�B )   1     +�
3 _    7     ��9 �   +     e>� �    2     Ŵ�� �    /     �]Y� �    2     f�� y   l     c���     I      c����uv                                                                                                                                                                                                                                                                                                                                                             BTHD       d(�     
 
       ��zW��QFHDB J�          ��)o     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      ��     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �     i�      �     i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������r                                                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FSSE        �   
  �   �  y   �D>�OHDR�    �      �          ?      @ 4 4�     +         �                   	 z�     �     2�         ��     �      5   ���������                                                      D                                                                       7                                                          2                                                     /                                                  e�WFSSE ��     � N    vs5        �XMTREE  ����������������                                       y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTLF 	     I       _    7      �    2      �    /      �    2      )   1      Z         y   l      �   +     ���#                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�                      ?      @ 4 4�     +         �                   f�               
 �     ��     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              �D�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 #   +  	   e>� t   2     Ŵ�� E   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c���n��I                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    A         2      E   /      t   2      �   1      �   L      #   +  	   z.��                                                                                                                                                                                                                                                                                                                                                                                    OCHK< !   0   REFERENCE_LIST 6     dataset        dimension                         ��             ��              ��             �            �            �J            �p            �            �            w�                        �             �7            P�             �            ��            ��             �             T�             ��             ��             D�             ~�            ��             ��             9             Zt            �b            �*            ��             $f�^FHDB �          �\ז     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                  FHDB R}          +���     _Netcdf4Coordinates                                   
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         dtype          bool     DIMENSION_LIST                              i�      �     i�      �     i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 TREE  ����������������#                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OCHK    ��     @          0   REFERENCE_LIST 6     dataset        dimension                         ��             �            �            Aa            �p            �            �                        Zt             �b             2�             ,�             �׵jFRHP               ���������      �                           	                                                      (  p�       R.�BTHD 	      d(U�     	 	       38
      2�             �Ka�TREE  ����������������#                       �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             FSHD  �                             P x (        X�                   �� �BTLF  W    D     �$ e   1     +�
3 �    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   L     c���     A      c���ȿZ]                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    D      �    7      �    2         /      3   2      e   1      �   L      �   +     W��                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�    �      �          ?      @ 4 4�     +         �                   ��     �         	 ��     s�     j�      �                                                                    D                                                                       7                                                          2                                                     /                                                  2                                                     1                                                    L                                                                               �t�FHDB z�          �\�$     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             FSHD  �                             P x (        U�                   j�KGBTLF  W    D     �$ e   1     +�
3 �    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   L     c���     A      c���ȿZ]                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    D      �    7      �    2         /      3   2      e   1      �   L      �   +     W��                                                                                                                                                                                                                                                                                                                                                                                                 FSSE �     �     �D�FHDB ��          fo�{     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              i�      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   