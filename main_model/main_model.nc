�HDF

         ��������G     0       ��OHDRJ"     .       ��      u1     �1     
          �      	      >	       �                                                                                                                                                                                                                                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                -�"�FRHP              �      �0      d	       @              (             �                                            (  ��       >7\�BTHD       d(�	              Io�BTHD 	      d(�              ���FSHD  P	                            P x (        Q�      @       @       ~\��BTLF  �'   R  
   �d  {    <     ��          j�    O     >��  �&   @     �*% s   E     l3�- B   1     +�
3 H    3     ��!6    6    �Q@     y    kW�G L   (     Ce`     2      @�k �&   R     �s� �    (     ʻ#� P   R     kϬ� �&   $     ���    .     Ŵ�� �   2     �]Y� �   r     f�� E'   O  	   1��� �    "     ��N� t   @     K���WA
                                                                                                                                BTLF 	     2       H    3      {    <      �    (      �    "          y     �&   @      �&   $      �&   R      E'   O  	    �'   R  
       O      P   R         6               L   (      t   @      �   r      �   2         .      B   1      s   E     �;;�                                                                                                                                                                                                                        BTHD       d(��              I�q&     applied_math    ��     history:
- plan
- custom_constraints_state.yml
- custom_constraints_delay.yml
data:
  variables:
    flow_cap:
      title: Technology flow (a.k.a. nominal) capacity
      description: A technology's flow capacity, also known as its nominal or nameplate
        capacity.
      default: 0
      unit: power
      foreach:
      - nodes
      - techs
      - carriers
      bounds:
        min: 0
        max: flow_cap_max
    link_flow_cap:
      title: Link flow capacity
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
      title: Carrier outflow
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
      title: Carrier inflow
      description: The inflow to a technology per timestep, also known as the flow
        consumed (by `storage` technologies) or the flow sent (by `transmission` technologies)
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
      title: Carrier export
      description: The flow of a carrier exported outside the system boundaries by
        a technology per timestep.
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
      title: Area utilisation
      description: The area in space utilised directly (e.g., solar PV panels) or
        indirectly (e.g., biofuel crops) by a technology.
      default: 0
      unit: area
      foreach:
      - nodes
      - techs
      where: (area_use_min OR area_use_max OR area_use_per_flow_cap OR sink_unit=per_area
        OR source_unit=per_area)
      bounds:
        min: 0
        max: area_use_max
    source_use:
      title: Source flow use
      description: The carrier flow consumed from outside the system boundaries by
        a `supply` technology.
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
      title: Source flow capacity
      description: The upper limit on a flow that can be consumed from outside the
        system boundaries by a `supply` technology in each timestep.
      default: 0
      unit: power
      foreach:
      - nodes
      - techs
      where: base_tech=supply
      bounds:
        min: 0
        max: source_cap_max
    storage_cap:
      title: Stored carrier capacity
      description: The upper limit on a carrier that can be stored by a technology
        in any timestep.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      where: include_storage=True OR base_tech=storage
      domain: real
      bounds:
        min: 0
        max: storage_cap_max
      active: true
    storage:
      title: Stored carrier
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
      title: Number of purchased units
      description: "Integer number of a technology that has been purchased,\nfor any
        technology set to require integer capacity purchasing.\nThis is used to allow
        installation of fixed capacity units of technologies (\nif `flow_cap_max`
        == `flow_cap_min`) and/or to set a fixed cost for a technology,\nirrespective
        of its installed capacity.\nOn top of a fixed technology cost,\na continuous
        cost for the quantity of installed capacity can still be applied.\n\nSince
        technology capacity is no longer a continuous decision variable,\nit is possible
        for these technologies to have a lower bound set on outflow/consumption\n
        which will only be enforced in those timesteps that the technology is operating.\n
        Otherwise, the same lower bound forces the technology to produce/consume\n
        that minimum amount of carrier in *every* timestep.\n"
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
      title: Number of operating units
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
      title: Available carrier flow capacity
      description: Flow capacity that will be set to zero if the technology is not
        operating in a given timestep and will be set to the value of the decision
        variable `flow_cap` otherwise. This is useful when you want to set a minimum
        flow capacity for any technology investment, but also want to allow the model
        to decide the capacity. It is expected to only be used when `purchased_units_max
        == 1`, i.e., the `purchased_units` decision variable is binary. If `purchased_units_max
        > 1`, you may get strange results and should instead use the less flexible
        `flow_cap_per_unit`.
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
      title: Asynchronous carrier flow switch
      description: Binary switch to force asynchronous outflow/consumption of technologies
        with both `flow_in` and `flow_out` defined. This ensures that a technology
        with carrier flow efficiencies < 100% cannot produce and consume a flow simultaneously
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
      title: Unmet demand (load shedding)
      description: Virtual source of carrier flow to ensure model feasibility. This
        should only be considered a debugging rather than a modelling tool as it may
        distort the model in other ways due to the large impact it has on the objective
        function value. When present in a model in which it has been requested, it
        indicates an inability for technologies in the model to reach a sufficient
        combined supply capacity to meet demand.
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
      title: Unused supply (curtailment)
      description: 'Virtual sink of carrier flow to ensure model feasibility. This
        should only be considered a debugging rather than a modelling tool as it may
        distort the model in other ways due to the large impact it has on the objective
        function value. In model results, the negation of this variable is combined
        with `unmet_demand` and presented as only one variable: `unmet_demand`. When
        present in a model in which it has been requested, it indicates an inability
        for technologies in the model to reach a sufficient combined consumption capacity
        to meet required outflow (e.g. from renewables without the possibility of
        curtailment).'
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
    ship_depart:
      description: 1 if a ship departs with an aluminium shipment from Iceland in
        this timestep, else 0
      foreach:
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      domain: integer
      bounds:
        min: 0
        max: 1
    ship_available:
      description: Number of ships available at origin (Iceland) at a given timestep
      foreach:
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      domain: integer
      bounds:
        min: 0
        max: 1
  global_expressions:
    flow_out_inc_eff:
      title: Carrier outflow including losses
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
      title: Carrier inflow including losses
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
    cost_operation_variable:
      title: Variable operating costs
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
        - where: any(carrier_export, over=carriers) AND any(cost_export, over=carriers)
          expression: sum(cost_export * flow_export, over=carriers)
        - where: NOT (any(carrier_export, over=carriers) AND any(cost_export, over=carriers))
          expression: '0'
        cost_flow_in:
        - where: base_tech=supply
          expression: cost_flow_in * source_use
        - where: NOT base_tech=supply
          expression: sum(cost_flow_in * flow_in, over=carriers)
        cost_flow_out:
        - expression: sum(cost_flow_out * flow_out, over=carriers)
    cost_investment_flow_cap:
      title: Flow capacity investment costs
      description: The investment costs associated with the nominal/rated capacity
        of a technology.
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
      title: Storage capacity investment costs
      description: The investment costs associated with the storage capacity of a
        technology.
      default: 0
      foreach:
      - nodes
      - techs
      - costs
      where: cost_storage_cap AND storage_cap
      equations:
      - expression: cost_storage_cap * storage_cap
    cost_investment_source_cap:
      title: Source flow capacity investment costs
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
      title: Area utilisation investment costs
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
      title: Binary purchase investment costs
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
      title: Total investment costs
      description: The installation costs of a technology, including those linked
        to the nameplate capacity, land use, storage size, and binary/integer unit
        purchase.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment_flow_cap OR cost_investment_storage_cap OR cost_investment_source_cap
        OR cost_investment_area_use OR cost_investment_purchase
      equations:
      - expression: sum(default_if_empty(cost_investment_flow_cap, 0), over=carriers)
          + default_if_empty(cost_investment_storage_cap, 0) + default_if_empty(cost_investment_source_cap,
          0) + default_if_empty(cost_investment_area_use, 0) + default_if_empty(cost_investment_purchase,
          0)
    cost_investment_annualised:
      title: Equivalent annual investment costs
      description: An annuity factor has been applied to scale lifetime investment
        costs to annual values that can be directly compared to operation costs. If
        the modeling period is not equal to one full year, this will be scaled accordingly.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment
      equations:
      - expression: $annualisation_weight * $depreciation_rate * cost_investment
      sub_expressions:
        annualisation_weight:
        - expression: sum(timestep_resolution * timestep_weights, over=timesteps)
            / 8760
        depreciation_rate:
        - where: cost_depreciation_rate
          expression: cost_depreciation_rate
        - where: NOT cost_depreciation_rate AND cost_interest_rate=0
          expression: 1 / lifetime
        - where: NOT cost_depreciation_rate AND cost_interest_rate>0
          expression: (cost_interest_rate * ((1 + cost_interest_rate) ** lifetime))
            / (((1 + cost_interest_rate) ** lifetime) - 1)
    cost_operation_fixed:
      title: Total fixed operation costs
      description: The fixed, annual operation costs of a technology, which are calculated
        relative to investment costs. If the modeling period is not equal to one full
        year, this will be scaled accordingly.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment AND (cost_om_annual OR cost_om_annual_investment_fraction)
      equations:
      - expression: "$annualisation_weight * (\n  sum(cost_om_annual * flow_cap, over=carriers)
          +\n  cost_investment * cost_om_annual_investment_fraction\n)"
      sub_expressions:
        annualisation_weight:
        - expression: sum(timestep_resolution * timestep_weights, over=timesteps)
            / 8760
    cost:
      title: Total costs
      description: The total annualised costs of a technology, including installation
        and operation costs.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment_annualised OR cost_operation_variable OR cost_operation_fixed
      equations:
      - expression: default_if_empty(cost_investment_annualised, 0) + $cost_operation_sum
          + default_if_empty(cost_operation_fixed, 0)
      sub_expressions:
        cost_operation_sum:
        - where: cost_operation_variable
          expression: sum(cost_operation_variable, over=timesteps)
        - where: NOT cost_operation_variable
          expression: '0'
      active: true
  constraints:
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
      description: Set a technology's area use to zero if its flow capacity upper
        bound is zero.
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
      description: Set an upper bound on the total area that all technologies with
        `area_use` can occupy at a given node.
      foreach:
      - nodes
      where: area_use AND available_area
      equations:
      - expression: sum(area_use, over=techs) <= available_area
    flow_capacity_systemwide_max:
      description: Set an upper bound on flow capacity of a technology across all
        nodes in which the technology exists.
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
        the total production of a given carrier to equal the total consumption of
        that carrier at every node in every timestep.
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
      description: Fix the outflow of a `supply` technology to its consumption of
        the available source.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: base_tech=supply AND NOT include_storage=True
      equations:
      - expression: flow_out_inc_eff == source_use * source_eff
    balance_supply_with_storage:
      description: Fix the outflow of a `supply` technology to its consumption of
        the available source, with a storage buffer to temporally offset the outflow
        from source consumption.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: storage AND base_tech=supply
      equations:
      - expression: storage == $storage_previous_step + source_use * source_eff -
          flow_out_inc_eff
      sub_expressions:
        storage_previous_step:
        - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
          expression: storage_initial * storage_cap
        - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
            \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT cluster_first_timestep=True"
          expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1)
            * roll(storage, timesteps=1)
        - where: cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
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
      description: Fix the quantity of carrier stored in a `storage` technology at
        the end of each timestep based on the net flow of carrier charged and discharged
        and the quantity of carrier stored at the start of the timestep.
      foreach:
      - nodes
      - techs
      - timesteps
      where: (include_storage=true or base_tech=storage) AND NOT (base_tech=supply
        OR base_tech=demand)
      equations:
      - expression: "storage == $storage_previous_step -\n  sum(flow_out_inc_eff,
          over=carriers) + sum(flow_in_inc_eff, over=carriers)"
      sub_expressions:
        storage_previous_step:
        - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
          expression: storage_initial * storage_cap
        - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
            \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT cluster_first_timestep=True"
          expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1)
            * roll(storage, timesteps=1)
        - where: cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
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
      where: base_tech=transmission AND NOT techs=aluminium_transport_tech
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
      description: Set the lower bound of a technology's outflow to a technology's
        carrier export, for any technologies that can export carriers out of the system.
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
      description: Limit flow capacity to zero if the technology is not operating
        in a given timestep.
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
      description: Fix the storage capacity of any technology using integer units
        to define its capacity.
      foreach:
      - nodes
      - techs
      where: storage AND purchased_units AND storage_cap_per_unit
      equations:
      - expression: storage_cap == purchased_units * storage_cap_per_unit
    flow_capacity_units_milp:
      description: Fix the flow capacity of any technology using integer units to
        define its capacity.
      foreach:
      - nodes
      - techs
      - carriers
      where: purchased_units AND flow_cap_per_unit
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
    flow_capacity_minimum:
      description: Set the lower bound on a technology's flow capacity, for any technology
        with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      - carriers
      where: flow_cap_min
      equations:
      - where: NOT purchased_units
        expression: flow_cap >= flow_cap_min
      - where: purchased_units
        expression: flow_cap >= flow_cap_min * purchased_units
    storage_capacity_max_purchase_milp:
      description: Set the upper bound on a technology's storage capacity, for any
        technology with integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: purchased_units AND storage_cap_max
      equations:
      - expression: storage_cap <= storage_cap_max * purchased_units
    storage_capacity_minimum:
      description: Set the lower bound on a technology's storage capacity for any
        technology with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: storage_cap_min
      equations:
      - where: NOT purchased_units
        expression: storage_cap >= storage_cap_min
      - where: purchased_units
        expression: storage_cap >= storage_cap_min * purchased_units
    area_use_minimum:
      description: Set the lower bound on a technology's area use for any technology
        with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: area_use_min
      equations:
      - where: NOT purchased_units
        expression: area_use >= area_use_min
      - where: purchased_units
        expression: area_use >= area_use_min * purchased_units
    source_capacity_minimum:
      description: Set the lower bound on a technology's source capacity for any supply
        technology with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: base_tech=supply AND source_cap_min
      equations:
      - where: NOT purchased_units
        expression: source_cap >= source_cap_min
      - where: purchased_units
        expression: source_cap >= source_cap_min * purchased_units
    unit_capacity_max_systemwide_milp:
      description: Set the upper bound on the total number of units of a technology
        that can be purchased across all nodes where the technology can exist, for
        any technology using integer units to define its capacity.
      foreach:
      - techs
      where: purchased_units AND purchased_units_max_systemwide
      equations:
      - expression: sum(purchased_units, over=nodes) <= purchased_units_max_systemwide
    unit_capacity_min_systemwide_milp:
      description: Set the lower bound on the total number of units of a technology
        that can be purchased across all nodes where the technology can exist, for
        any technology using integer units to define its capacity.
      foreach:
      - techs
      where: purchased_units AND purchased_units_max_systemwide
      equations:
      - expression: sum(purchased_units, over=nodes) >= purchased_units_min_systemwide
    async_flow_in_milp:
      description: Set a technology's ability to have inflow in the same timestep
        that it has outflow, for any technology using the asynchronous flow binary
        switch.
      foreach:
      - nodes
      - techs
      - timesteps
      where: async_flow_switch
      equations:
      - expression: sum(flow_in, over=carriers) <= (1 - async_flow_switch) * bigM
    async_flow_out_milp:
      description: Set a technology's ability to have outflow in the same timestep
        that it has inflow, for any technology using the asynchronous flow binary
        switch.
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
    shipping_capacity_link:
      description: Allow aluminium flow out of Iceland only when a ship departs (enforce
        discrete ship usage)
      foreach:
      - nodes
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      equations:
      - expression: flow_in[carriers=aluminium] <= ship_depart * flow_cap_max
    ship_limit:
      description: There's only one ship so you can only have one departure every
        144 hours
      foreach:
      - nodes
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      equations:
      - expression: sum_next_n(ship_depart, timesteps, 144) <= 1
    ship_availability_limit:
      description: You can only depart if there is availability
      foreach:
      - nodes
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      equations:
      - expression: ship_depart <= $prev_available
      sub_expressions:
        prev_available:
        - where: timesteps = get_val_at_index(timesteps=0)
          expression: '1'
        - where: timesteps >= get_val_at_index(timesteps=-72)
          expression: '0'
        - where: NOT timesteps = get_val_at_index(timesteps=0) AND NOT timesteps >=
            get_val_at_index(timesteps=-72)
          expression: roll(ship_available, timesteps=1)
    ship_roundtrip_cycle:
      description: "Ensures the ship returns to origin before another departure can
        occur. ship_available(t) = ship_available(t-1) + (returning ship at t) - (departing
        ship at t).\n"
      foreach:
      - techs
      - timesteps
      where: techs=aluminium_transport_tech AND nodes=Iceland
      equations:
      - expression: ship_available == $prev_available + $ship_returned - ship_depart
      sub_expressions:
        prev_available:
        - where: timesteps = get_val_at_index(timesteps=0)
          expression: '1'
        - where: NOT timesteps = get_val_at_index(timesteps=0)
          expression: roll(ship_available, timesteps=1)
        ship_returned:
        - expression: default_if_empty(roll(ship_depart, timesteps=144), default=0)
    aluminium_transport_delay:
      description: "Enforce a 72-hour delay for aluminium transport from Iceland to
        Netherlands. The amount arriving at NL at time t equals the amount sent from
        Iceland at time t - 72.\n"
      foreach:
      - techs
      - timesteps
      where: techs=aluminium_transport_tech
      equations:
      - expression: "flow_out[nodes=Netherlands, carriers=aluminium] == default_if_empty(\n\
          \     roll(flow_in[nodes=Iceland, carriers=aluminium], timesteps=72),\n\
          \     default=0\n   )\n"
  piecewise_constraints: {}
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
BTLF �      �             }I�W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OHDR(                                     *       ��      �       ��     Q            ������������������������A         _Netcdf4Coordinates                            +        CLASS          DIMENSION_SCALE          NAME          nodes   i���FRHP               ��������~      .@                                                                                 (  /�       |9��FSSE .@     ~ �    @�� �HK�FSSE �]     � d    bW�w        ��LAFSHD  �                             P x (        ��                   ��*BTHD       d(�     
 
       �SOHDR+                                     *       ��      �       9S     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE #        NAME    	      carriers   �2R�OHDR    �      �                @    *         �    Ͷ     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE $        NAME    
      timesteps   �~�ABTHD       d(D     
 
       �N:� ��FRHP               ���������       I�                                                                                  (  ��        �_�BTHD       d(��              2LdnBTHD 	      d(��              �l�                  GCOL                        applied_math                  config                defaults
              ;�      j               n              ��     �               �               �               �               �               �               �              fuel_consumption_tech   �              fuel_necessity  �              fuel_supply     �              aluminium_transport_tech�              aluminium_supply�              aluminium_demand�               �              ��      �              ��     �               �               �              Netherlands     �              Iceland �               �              ��      �              ��     �              }�      �               �               �              fuel    �       	       aluminium                    ��                   ��                  }�                   ��                   ��                   ��                  }�                   ��                   ��                   ��                  ��                   ��                  ��                   ��                  ��                   ��                   ��                  }�                    ��      !             ��      "             ��     #             }�      $             ��      .              /             ��      0             ��     1             ��      2             ��      3             ��      4             ��     5             ��      6              7             monetary>             ��      ?             ��     @             }�      A             ��      C             ��     D             }�      H             }�      I             ��     J             ��      L             }�      M             ��      N             ��      O              P             ��     Q              R              S              T              U              V              W             transmission    X             demand  Y             supply  Z             transmission    [             supply  \             demand  ]              _             ��      `             ��     a              b              c              d              e              f              g              h              i              j              k              l              m      	       aluminium       n              o              p              q             fuel    r              s              t              u             fuel    v              w      	       aluminium       x              |             ��      }             ��     ~             }�                    �             ��     �              �              �              �              �              �              �             #3B61E3 �             #87CEEB �             #E37A72 �             #8465A9 �             #F9CF22 �             #072486 �             ��     �              �             ��     �              �              �              �              �              �              �             Fuel Consumption�             Fuel Necessity  �             Fuel Supply     �             Aluminium Transport     �             Aluminium Supply�             Aluminium Demand�             ��      �             ��     �             }�      �             ��     �             default �      	       is_result       �             ��     �             ��      �             ��     �             ��      �             ��      �             ��      �             ��      �             ��      �             ��     �             ��      �             ��      �             ��     �             ��      �             ��     �             }�      �             ��     �             ��      �             ��              �                                                                                                                                               OHDR(                                     *       ��      .      �     Q           
 ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          costs   Ï�          ��c�OHDR4                                                  ?      @ 4 4�     +         �                   "�                       �      ��      ��       5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ��� FSHD  �                              P x (        ��                    ?`��BTLF  _    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    D     Q�ń T   +     e>� �   2  
   Ŵ�� S   2  	   �]Y� !   2     f�� V   5     bS�� �   l     c���     I      c��� �    o     G��� �   @     K���r                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    D      �    o      V   5      �         �   7      �   @      !   2      S   2  	    �   2  
    �   1      �   l      T   +     �A3�                                                                                                                                                                                                                                                                                                                                BTHD 	      d(�v     
 
       &T�BTHD 	      d(�     
 
       �@S�^JBTHD       d(cn             �k�GFSSE s�     : �    �
@FSSE �     �  P    ]�!�BTHD       d(�             �k�BTHD 	      d(�             �ޑ�FSSE I�      �      KX��FRHP               ���������      �1                          )                                                         ��      G0,j           ��FSSE d	        �    �'    H �    �              yTm�FSSE      �  M    ����FHDB �           Rf�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title    )      Technology flow (a.k.a. nominal) capacity     description    N      A technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �      ��      �        _Netcdf4Dimid                                                                                                                                                  FHDB �          ���     config         init:
  name: main model
  calliope_version: 0.7.0
  broadcast_param_data: true
  time_subset:
  - '2005-01-01'
  - '2005-01-31'
  time_resample:
  time_cluster:
  time_format: ISO8601
  distance_unit: km
build:
  mode: plan
  add_math:
  - custom_constraints_state.yml
  - custom_constraints_delay.yml
  ignore_mode_math: false
  backend: pyomo
  ensure_feasibility: false
  objective: min_cost_optimisation
  pre_validate_math_strings: true
  operate:
    window: 24h
    horizon: 48h
    use_cap_results: false
solve:
  save_logs:
  solver_io:
  solver_options: {}
  solver: gurobi
  zero_threshold: 1e-10
  shadow_prices: []
  spores:
    scoring_algorithm: integer
    number: 3
    save_per_spore_path:
    skip_baseline_run: false
    tracking_parameter:
    score_threshold_factor: 0.1
 	    def_path    
      model.yaml     serialised_single_element_list  ?      @ 4 4�         serialised_bools  ?      @ 4 4�                              FHDB �           ��j 	    defaults    [     bigM: 1000000000.0
objective_cost_weights: 1
spores_baseline_cost: .inf
spores_slack: 0
spores_score: 0
color: .nan
name: .nan
cap_method: continuous
integer_dispatch: false
include_storage: false
force_async_flow: false
flow_cap_per_storage_cap_min: 0
flow_cap_per_storage_cap_max: .inf
flow_cap: .inf
flow_cap_max: .inf
flow_cap_max_systemwide: .inf
flow_cap_min: 0
flow_cap_min_systemwide: 0
flow_out_min_relative: 0
flow_cap_per_unit: .nan
flow_in_eff: 1.0
flow_out_eff: 1.0
flow_out_parasitic_eff: 1.0
flow_ramping: 1.0
export_max: .inf
lifetime: .inf
area_use: .inf
area_use_max: .inf
area_use_min: 0
area_use_per_flow_cap: .nan
storage_cap: .inf
storage_cap_max: .inf
storage_cap_min: 0
storage_cap_per_unit: .nan
storage_discharge_depth: 0
storage_initial: 0
storage_loss: 0
cyclic_storage: true
purchased_units_min_systemwide: 0
purchased_units_max_systemwide: .inf
purchased_units: .inf
purchased_units_min: 0
purchased_units_max: .inf
sink_unit: absolute
sink_use_min: 0
sink_use_max: .inf
sink_use_equals: .nan
source_unit: absolute
source_cap_equals_flow_cap: false
source_eff: 1.0
source_use_min: 0
source_use_max: .inf
source_use_equals: .nan
source_cap: .inf
source_cap_max: .inf
source_cap_min: 0
one_way: false
distance: 1.0
flow_in_eff_per_distance: 1.0
flow_out_eff_per_distance: 1.0
cost_flow_cap_per_distance: 0
cost_purchase_per_distance: 0
cost_flow_cap: 0
cost_export: 0
cost_interest_rate: 0
cost_om_annual: 0
cost_om_annual_investment_fraction: 0
cost_flow_in: 0
cost_flow_out: 0
cost_purchase: 0
cost_area_use: 0
cost_source_cap: 0
cost_storage_cap: 0
cost_depreciation_rate: 1
available_area: .inf
     allow_operate_mode                                name    
      main model     timestamp_model_creation  ?      @ 4 4�                P�G�A     timestamp_build_start  ?      @ 4 4�                F�G�A     timestamp_build_complete  ?      @ 4 4�                ���H�A                          FHDB �           ���W     termination_condition          optimal     calliope_version_defined          0.7.0     calliope_version_initialised    
      0.7.0.dev6     applied_overrides            	    scenario          None     timestamp_solve_start  ?      @ 4 4�                ��H�A     timestamp_solve_complete  ?      @ 4 4�                �lI�A     serialised_dicts                               ��            ��            ��              serialised_nones          scenario     serialised_sets  ?      @ 4 4�         _NCProperties    "      version=2,netcdf=4.9.2,hdf5=1.14.4                                                                                                                                                                                                                                                                                                                                        FHIB �           ��      ��      ��������������������������������������������������      ������������������������T�d�TREE  ����������������-                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                OHDR�                      ?      @ 4 4�     +         �                   5                6     �     �      #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      y�DoFRHP               ���������                                                                                        (  5       Й"�BTHD       d(f             p͡�BTHD 	      d(f             �V��FSHD  �                              P x (        ��                    �%NBTLF  W    D     �$ �   1     +�
3 y        �I75 �   7     ��9 �    -     Q�ń �   +     e>� s   2  
   Ŵ�� A   2  	   �]Y�    2     f�� D   5     bS��    L     c���     A      c��� �    |     G��� �   @     K���Җ�]                                                                                                                                                                                                                                                                        BTLF 	     A       W    D      �    -      �    |      D   5      y         �   7      �   @         2      A   2  	    s   2  
    �   1      �   +         L     o �                                                                                                                                                                                                                                                                                                                                FRHP               ��������|       �                                                                                 (  �s       �$��           1�x^c`(P �0���ˁ��� �6�W_� � T� &� ��=FHDB 6          ��x�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      �     title          Link flow capacity     description    [      A transmission technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         _Netcdf4Dimid                     DIMENSION_LIST                              ��      n                                                                                                                                                                                      TREE  ����������������                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  OHDR]D                         �                           �          ?      @ 4 4�     +         �                   !"              �          �     H�      n�       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               �YQDFRHP               ��������*       `                                                                                 (  !       ��9FSHD  *                              P x (        c�                   ܁Y�BTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    *     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� h   2     f�� �   5     bS�� /   |     c���     M      c��� �    �     G��� (   @     K���>���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    *      �    �      �   5      �          �   7      (   @      h   2      �   2  	    �   2  
    �   1      /   |      �   +     x��                                                                                                                                                                                                                                                                                                                                BTHD       d(�             :
��BTHD 	      d(�             i^�FSHD  '                             P x (        ��                    �nFBTHD       d(ś             Ƒa�         ��MBTHD 	      d(cp             �4�FSHD  |                              P x (        �T                   �*�      X�O/x^c`�
��Ǐ�2?~��;8�� <4�FHDB �          �5G     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier outflow     description    �      The outflow of a technology per timestep, also known as the flow discharged (from `storage` technologies) or the flow received (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��           ��             _Netcdf4Dimid                                                           TREE  ����������������.                                              �=                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR D                         �                           �          ?      @ 4 4�     +         �                   �D              �          �8     %9     K9                                         ��BTHD       d(2     ) )       ��޾BTHD      d(�f      )       0���FSHD  �                              P x          ��     (       (       �V#rBTLF �g \  ! i�8 �   ���    �J �   y}W !   ���  *  # �rQ' �    5�. *   �8 �    ���I �  ) )m�M   & y��P |    `��P    ��-S {  , w�iV z   ���X i   ���[ }  # �^P^ N   _.]c >   ��l �     ��{� b   q���    1M7� ,  " D9� �   �ԕ� �   ��_� `    J鱷 �  ' 7��� N  - ĕ�� �    ho�     ��� �   �L�� /   S�:� D   Ѧ� -    ��� �  * d�� E    �;�� �  " yܴ� �  ! �}"�    XX� �  $ ��� �   �M�=                                                   BTLF         �            �             �            �    	           
        !           >           \  !         }  #         �  "         �  *                    -                        E            `    ��K                                                                                                                                                                                                                                                                      FRHP               ��������:      s�                                                                                 (  ,�       &g�FSSE �U     �  n    �J��FSSE cr     �  m    1f�FRHP               ���������       �                                                                                 (  ��       �'D      Yc�\OHDR 4                  �                    �         +         �                   _           �          �T     NU     tU         ?      @ 4 4�   3                                                       ��6                FRHP               ��������4       q9                                                                                 (  �>       2wIKBTHD       d(�9             "`BTHD 	      d(�;             ��!FSHD  4                              P x (        Ϸ                   ��XBTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    )     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� ^   2     f�� �   5     bS�� %   |     c���     M      c��� �    �     G���    @     K���	;Lb                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    )      �    �      �   5      �          �   7         @      ^   2      �   2  	    �   2  
    �   1      %   |      �   +     8�                                                                                                                                                                                                                                                                                                                                x^�ڱ�@��a4R6�8Y�e�1`$���+Y� \��x                                                                                        �ye�N�C�
��        ���ݫith\���    ���X�;?��4:4���J_r�     ��?�r�  @���:k޽�F���_��= ה��=  �g`	�?��U�*�n�A��`��=     �m7��  ��]�Y�A�ʠ]eЭ2�W�H��k�wFXQ�  ����N FHDB �8          ^]C     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow     description    �      The inflow to a technology per timestep, also known as the flow consumed (by `storage` technologies) or the flow sent (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��           ��             _Netcdf4Dimid                                                                     FHDB ��          V9S�       nodes��             techs��            carriers}�             	timesteps��             costs��             flow_cap;�             link_flow_capb�             flow_oute            flow_ini0     	       
source_usew7     
       
source_capTk            ship_depart#�            ship_availablef�            flow_out_inc_effd�            flow_in_inc_effv�            cost_operation_variable��                         TREE  ����������������(                                              �Y                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OCHK                 +        _Netcdf4Dimid                   ��     *       V                                                                                          ���2          O��@FRHP               ��������'      �                                                                                 (  .�       #Sf�FSSE �     |  �    7GM�    FRHP               ���������       �U                                                                                 (  [       �	MBTHD       d(�U             ���BTHD 	      d(�W              �7<FSHD  �                              P x (        �6                   K�/]BTLF  _    D     �$ �   1     +�
3 y         �I75 �   7     ��9 �    *     Q�ń C   +     e>� t   2  
   Ŵ�� B   2  	   �]Y�    2     f�� D   5     bS�� �   l     c���     I      c��� �    w     G��� �   @     K����X                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    *      �    w      D   5      y          �   7      �   @         2      B   2  	    t   2  
    �   1      �   l      C   +     O��                                                                                                                                                                                                                                                                                                                                x^�ڱB1Pa���Ê��\�dGy���+"+�                                                                                         =2h��ҸB#+    �0�~��ҸB#+       ��  ��|���hi\����������  ���y�S����_��=     �"g����'  0F�G�v�A�ʠ[eЯ2&w��
��  ���ʠ]eЮ2�V��~$w     ���r� ���?N FHDB �T          ��Y^     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Source flow use     description    V      The carrier flow consumed from outside the system boundaries by a `supply` technology.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��             _Netcdf4Dimid                                                                                                                                                                   TREE  �����������������                                       Es                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR$                                    ?      @ 4 4�     +         �                   �w                   f     ��      �      J                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             �hS�BTLF  [    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    /     Q�ń Y   +     e>� �   2  
   Ŵ�� h   2  	   �]Y� 6   2     f�� k   5     bS�� �   \     c���     E      c��� �    �     G��� �   @     K����闯                                                                                                                                                                                                                                                                        BTLF 	     E       [    D      �    /      �    �      k   5      �         �   7      �   @      6   2      h   2  	    �   2  
    �   1      �   \      Y   +     ���                                                                                                                                                                                                                                                                                                                                FSHD  �                              P x (        �6                   �e�OCHK   (�     R      +        _Netcdf4Dimid                  j-bFSSE {�     �  R    f��FSSE �     �  i    )��e     �Ik&x^�ڡ� ��a<R7�8]�0�"��������                                           ��A;��+4�      �
?X���o�ѣq�FV��� ��r�    �s��G�v�A�ʠ[eЯ2&w 0U��{    ޹6y�	FHDB f          �\�     _Netcdf4Coordinates                                    _FillValue  ?      @ 4 4�                      �     title          Source flow capacity     description    |      The upper limit on a flow that can be consumed from outside the system boundaries by a `supply` technology in each timestep.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �        _Netcdf4Dimid                                                                                                                                             TREE  ����������������                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�$           �             �          ?      @ 4 4�     +         �                   .�        �          T     �     �      5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ($�BTLF  [    D     �$ !   1     +�
3    7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� �   2     f�� R   \  	   c���     E      c��� �    u     G��� K   @     K���ٱ�                                                                                                                                                                                                                                                                                                                           BTLF 	     E       [    D      �    u         7      K   @      �   2      �   2      �   2      !   1      R   \  	    �   +  
   +�m�                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������       {�                                                                                 (  .�       Ns�BTHD       d(Ͳ             O`�VBTHD 	      d(ʹ             �Z��#�             _���x^c`�  �?����C��;�׃1  �iZFHDB T          �q�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      �     description    T      1 if a ship departs with an aluminium shipment from Iceland in this timestep, else 0 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��             _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                       TREE  ����������������e                               Ǡ                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�$           �             �          ?      @ 4 4�     +         �                   ,�        �          6     P     M�      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     $BTHD 	      d(ŝ             �U�FSHD  :                             P x (        �                    ��BTLF  [    D     �$    1     +�
3    7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� x   2     f�� ?   \  	   c���     E      c��� �    b     G��� 8   @     K����t�                                                                                                                                                                                                                                                                                                                           BTLF 	     E       [    D      �    b         7      8   @      x   2      �   2      �   2         1      ?   \  	    �   +  
   ���                                                                                                                                                                                                                                                                                                                                                                       BTHD       d(��             /���BTHD 	      d(��             TjD�FSHD  �                              P x (        ,�                    �e�FSSE `     *  �    +���BTHD 	      d(��             i3sFSSE �      � x    ,F    X,��x^�ױ !A:��h��Cb�r�f*��d�c                          �� ���N�U\Pdҡ�  ����>��)���L:�� �� P_FHDB 6          �i�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      �     description    A      Number of ships available at origin (Iceland) at a given timestep 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��             _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                          TREE  ����������������m                               ��                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDRD                         �                           �          ?      @ 4 4�     +         �                   .�              �          �     ��     Չ      *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ���FSHD  �                              P x (        s                   �BTLF  c    D     �$ z   1  
   +�
3 m   7     ��9 �    ;     Q�ń '   +     e>� H   2  	   Ŵ��    2     �]Y� �   2     f�� 8   5     bS�� �   |     c���     M      c��� �    V     G��� �   @     K���!���                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    ;      �    V      8   5      m   7      �   @      �   2         2      H   2  	    z   1  
    �   |      '   +     5���                                                                                                                                                                                                                                                                                                                                             OCHK   �     R      +        _Netcdf4Dimid                :        units          hours since 2005-01-01 00:00:001    	    calendar          proleptic_gregorian  �JϔBTHD 	      d(F     
 
       7�]j��            v��FSSE q9     4  �    �A� �{�YFRHP               ���������       cr                                                                                 (  A�       �-BTHD       d(��             8jY�      d�            %���x^�ױ	 A���Ҿe�?��c&�H<                         x��	yߧug2m�DL���{ ��> ��괪L�-�(�i����{ ��`f�FHDB �          P�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title           Carrier outflow including losses     description    5      Outflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��           ��              _Netcdf4Dimid                                                                                                                                                                                               TREE  ����������������.                                              o�                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR	D                         �                           �          ?      @ 4 4�     +         �                   ��              �          �6     ş     �      (                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           @��sBTLF  c    D     �$ x   1  
   +�
3 k   7     ��9 �    :     Q�ń %   +     e>� F   2  	   Ŵ��    2     �]Y� �   2     f�� 6   5     bS�� �   |     c���     M      c��� �    U     G��� �   @     K�����o�                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    :      �    U      6   5      k   7      �   @      �   2         2      F   2  	    x   1  
    �   |      %   +     �5��                                                                                                                                                                                                                                                                                                                                             OCHK   ��     �      +        _Netcdf4Dimid                   ��BTHD 	      d(!`     
 
       �A�WFSSE j�     t �    
9OHDR>4                                                 +         �                                         �     y     �         ?      @ 4 4�   q                                                                                                                     ��0�    Å�Nx^�ڱ�@��a4R6�8Y�e�1`$���+Y� \��x                                                                                        �ye�N�C�
��        ���ݫith\���    ���X�;?��4:4���J_r�     ��?�r�  @���:k޽�F���_��= ה��=  �g`	�?��U�*�n�A��`��=     �m7��  ��]�Y�A�ʠ]eЭ2�W�H��k�wFXQ�  ����N FHDB �6          �#j�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow including losses     description    4      Inflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      !     ��      "     ��      #     ��      $       _Netcdf4Dimid                                                                                                                                                                                                 TREE  ����������������(                                              �                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�D                         �                           �          ?      @ 4 4�     +         �                   A�              �          �     ��     ~�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            2B{�BTLF  c    D     �$ �   1     +�
3 a   '     �I75 �   7     ��9 �    3     Q�ń B   +     e>� c   2  
   Ŵ�� 1   2  	   �]Y� �   2     f�� ,   5     bS�� �   |     c���     M      c��� �    R     G��� �   @     K���/���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    3      �    R      ,   5      a   '      �   7      �   @      �   2      1   2  	    c   2  
    �   1      �   |      B   +     ]Ě                                                                                                                                                                                                                                                                                                                                OHDR(                                     *       ��      j       �r     Q           ! ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          techs   �^b�OCHK   @     �       +        _Netcdf4Dimid                  �h�FSHD  t                             P x (        �                   @��        ��wFHIB ��          �B          |E     ���������w��FSSE �     ' �    r��Cx^�ڱB1Pa���Ê��\�dGy���+"+�                                                                                         =2h��ҸB#+    �0�~��ҸB#+       ��  ��|���hi\����������  ���y�S����_��=     �"g����'  0F�G�v�A�ʠ[eЯ2&w��
��  ���ʠ]eЮ2�V��~$w     ���r� ���?N FHDB �          m#     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Variable operating costs     description    1      The operating costs per timestep of a technology.     default                                 unit          cost_per_time 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      /     ��      0     ��      1     ��      2       _Netcdf4Dimid                                                                                                                                                                    TREE  ����������������                                                                                 �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTHD 	      d(��     	 	       �˽�FSHD  �                             P x (                           Km�eFSSE �     �     ���FSSE ��     � u    r�FSSE ��     � J    _�!BTHD       d(�     
 
       |�t�         BTHD       d(�     
 
       &<T�jBTHD       d(!^     
 
       ME��                      FRHP               ���������       �                                                                                 (         �n��BTHD       d(             `��@BTHD 	      d(	             ���FSHD  �                              P x (        !s                   �:�BTLF  _    D     �$ �   1     +�
3 v        �I75 �   7     ��9 �    &     Q�ń >   +     e>� o   2  
   Ŵ�� =   2  	   �]Y�    2     f�� A   5     bS�� �   l     c���     I      c��� �    x     G��� �   @     K���k�                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    &      �    x      A   5      v         �   7      �   @         2      =   2  	    o   2  
    �   1      �   l      >   +     su�                                                                                                                                                                                                                                                                                                                                x^�ءQE1�/)�A⩀���(Ci��hּ�p�ϑ�V����r  �����      �}|@�{       `��@�       ��4��       �q��      �}�C����3(�ѡ�
-Y	   ~�������r����M� `UO�@���zˠ�F��+�w�� �����5z�F~���T�,&w ��Gh ��g���������~I�w �UMh w��{>��FHDB �          ��s     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Total costs     description    W      The total annualised costs of a technology, including installation and operation costs.     default                                 unit          cost 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      3     ��      4     ��      5       _Netcdf4Dimid                                                                                                                                                                        FHDB ��         ���       cost�            capacity_factor�@            systemwide_capacity_factor[            systemwide_levelised_cost�q            total_levelised_cost!b            bigMP�            objective_cost_weights��            	base_tech_�            carriero�            carrier_out��            color��             flow_cap_max!�     !       name�     "       
carrier_in,�     #       flow_out_eff%�                         TREE  ����������������:                                       W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         x^c` �E 
\��ې؟���� ��$"~8����,���ޡ���� �8P ��E   ��      �      ��      �      ��      �      ��      �      ��      �      ��      �      ��      �      ��      �   	   ��      �      ��      �      ��      7  x^�ء�0Aw���I	�P� ��](t@@�1                                                                                                         �<u���V\Pd     �x�u� �����ZqA�I �F8Q�{    ��u���V\Pd     Иp���  ���}E����$ ��?#���=  �����ax^c`hB�B�B��0�a�a��!������A�� �c
x^c�f��?"p�J�up�Gw8��C8ۗ�
�N������a� ���x^{�����Kz7�}��  S	�    ��.Ax^c` �� '0   ��      \     ��      [     ��      Z     ��      W     ��      X     ��      Y      ��      x  	   ��      w      ��      v      ��      s      ��      t     ��      u  	   ��      m      ��      n      ��      o      ��      p     ��      q      ��      r  x^c`\�Ǐ?� �� ����� ̰�� ��"	   ��      �     ��      �     ��      �     ��      �     ��      �     ��      �  x^c`�
��Ǉ�2?>��;8�� ;��   ��      �     ��      �     ��      �     ��      �     ��      �     ��      �  x^c`\�| #(e��=A	( ��"	x^c` ~����Ǐ�z{��z ?{;x^c` ~����Ǐ�z{��z ?{;x^c�f``�� "@��^��lw �e?b_(�
�� �)?~�p� $>��������$�� �B/x^{i��h��Ѳ��C?��� (y`x^[�p�����d�3O�
p  G-x^�Ʊ	   �L�.��C[
!3H�;x�                                  �mU�{���{���{���{���w�Sy���{���{���{���{��o���{���{���{���{���~�Yy���{���{���{���{���L�/x^�Ʊ	   A'r3�C��$�<|k                                  ��|y���{���{���{���{_���{���{���{���{������=�vx���{���{���{���{_�cD����{���{���{���{��!��/x^c``Bb` $  | 	x^c`x����@��� r�BYW"�La`��c��?����+�Mx^��1  �5�#X�o@   ���JOp�x^��1  �5�#X�o@   ���JOp�                                                                                                                                                                                                                                                                                                                                                                                           OHDR 4    �                    �                       +         �                   "�     �                �T     ��     _U         ?      @ 4 4�   3                                                       �T��  ϘFRHP               ���������      �]                           
                                                      (  �c       �T�S    ��            ��>�                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      FSSE ƌ     � d    ���BTHD 	      d(]�             ���FSHD  ~                             P x (        ��                    z�q�          k�OHDR�D                         �                           �          ?      @ 4 4�     +         �                   �L              �         
 uC     ��      ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )� �FRHP               ��������t      j�                           
                                                      (  �H       � BTLF  c    D     �$ �   1     +�
3 �    7     ��9 a   +  	   e>� �   2     Ŵ�� P   2     �]Y�    2     f�� �   |     c���     M      c��� �    @     K������                                                                                                                                                                                                                                                                                                                                            BTLF 	     M       c    D      �    7      �    @         2      P   2      �   2      �   1      �   |      a   +  	   #��                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������      �                            
                                                      (  �z       M �`BTHD       d(�t     
 
       Ň�            &"��FHDB uC          "G     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      >     ��      ?     ��      @     ��      A       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                     TREE  �����������������                                               A                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�$                                    ?      @ 4 4�     +         �                   �g                  
 (     �     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    �z�FSHD  �                             P x (        ݿ                    ���XBTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y�    2     f�� �   \     c���     E      c��� �    @     K���ۿ��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    @         2      H   2      z   2      �   1      �   \      9   +  	   �2k|                                                                                                                                                                                                                                                                                                                                                                                    OHDRH$                                   +         �                   �                  
 4�     W�      ��          ?      @ 4 4�   �                                                                                                                                                   G1�'          ;��FHDB (               _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      C     ��      D       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������3                               .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�4                                                  ?      @ 4 4�     +         �                   �~                     
 H     �H     ��       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        c�q�BTLF  _    D     �$ �   1     +�
3 �    7     ��9 M   +  	   e>� ~   2     Ŵ�� L   2     �]Y�    2     f�� �   l     c���     I      c��� �    @     K�����1�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   2      ~   2      �   1      �   l      M   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    FSSE �     � u    	���   ��KFRHP               ���������      �y                           	                                                      (  ��       �~|        Ə��BTHD       d(]�             ,�OBTHD       d(o�     	 	       A][<BTHD 	      d(o�     	 	       !�FSHD  �                             P x (        z                   f�]FSSE �y     � 	    ���FRHP               ���������      �                           	                                                      (  U�       �|�BTHD       d(��     	 	       4!�JcJFHDB H          K�     _Netcdf4Coordinates                                      _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      H     ��      I     ��      J       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                        TREE  ����������������1                                       a                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR                       ?      @ 4 4�     +         �                    �
     �     -�         �2            [                                                                                               �۳z   FRHP               ���������      ƌ                           
                                                      (  �       �j�xFSHD  �                             P x (        �?                   =�G�BTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y�    2     f�� �   \     c���     E      c��� �    @     K���ۿ��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    @         2      H   2      z   2      �   1      �   \      9   +  	   �2k|                                                                                                                                                                                                                                                                                                                                                                                    FHDB 4�          MrB     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      L     ��      M       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDRb           ?      @ 4 4�     *         �    �                  ������������������������D         _FillValue  ?      @ 4 4�                      �7    
    is_result                            A        default  ?      @ 4 4�                    e��A@        serialised_single_element_list  ?      @ 4 4�    2        serialised_dicts  ?      @ 4 4�    /        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    :��OHDR�                      ?      @ 4 4�     +         �                   /�                +�      Ky     @      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 D�v	BTLF  W    D     �$ �   1     +�
3 �    7     ��9 W   +  
   e>� �   2     Ŵ�� y   /     �]Y� G   2     f�� �    5     bS��    L  	   c���     A      c���    @     K�����                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    5         @      G   2      y   /      �   2      �   1         L  	    W   +  
   &��                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������      �j                           
                                                      (  �n       /�jBTHD 	      d(�l     
 
       G��      ��             ؂��FHDB +�           ��P�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      N       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                              TREE  ����������������                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OHDR�                                     *       ��      O      �      `           	 �x     qy     �y      �                                                                    7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    L                                                                               sNbaOHDR`4                                                 +         �                   U�                     
 ��     u�     ��         ?      @ 4 4�   �                                                                                                                                                       4��  _�             ����BTLF  a   1     +�
3 W    7     ��9 �   +     e>� /   2     Ŵ��     /     �]Y� �    2     f�� �   L     c���     A      c��� �    @     K����;��                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    7      �    @      �    2          /      /   2      a   1      �   L      �   +     ��|                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                                   *       ��      ]       !     �           	 *z     �z     �                                                                              7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    \                                                                                               ]�c�OHDRm                      ?      @ 4 4�     +         �                    H�     ��     ��         ��            �                                                                                                                                                                                                           ɟ9\BTHD 	      d(,�             �{T*pMOHDR                                     *       ��            
 $�     �     ��          "     `        ~                                                                                                                                 ��E    ���FHDB �x          �A,S     _Netcdf4Coordinates                            
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      P       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       BTLF  e   1     +�
3 [    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   \     c���     E      c��� �    @     K���e���                                                                                                                                                                                                                                                                                                                                                             BTLF 	     E       [    7      �    @      �    2         /      3   2      e   1      �   \      �   +     3it.                                                                                                                                                                                                                                                                                                                                                                                                 FSSE b�     � J     ���FSSE ��     r �    �I�FSSE �(     r �    Q�`X;[  BTHD 	      d(�     
 
       ��FSHD  �                             P x (        ��                   \1�	    BTHD 	      d(�     
 
       2\�     FRHP               ���������      ��                           
                                                      (  U�       D�BTHD       d(�     
 
       2���BTHD 	      d(�     
 
       BE^FSHD  �                             P x (                           '�+�BTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    OHDR                       ?      @ 4 4�     +         �                    �     ��     �T         z�            [                                                                                               ��ȃ         {[�BTHD 	      d(+             ���G��FHDB ��          ^���     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      |     ��      }     ��      ~       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            FHDB *z          Y�r     _Netcdf4Coordinates                                
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      _     ��      `       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                                        �!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FSHD  V                             P x (        �                   ��]                      BTHD       d(,�             ���o FRHP               ���������      ��                           
                                                      (  ��       W��FSHD  �                             P x (        7                   aA�lBTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    OHDR�                                     *       ��      �      
      R     <�         {"     `                                                                                                                                                                                                                                                                                     ����       ��fFHDB $�          )$��     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      BTHD       d(>P             ����BTHD 	      d(>R             ���t                BTHD       d(z;             �l�            FRHP               ��������r      ��                                                                                 (  ��       �T�FSHD  r                             P x (         �                   yӥ�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OHDR�4                                                 +         �                   t                     
      �     �         ?      @ 4 4�   �                                                                                                                                                                                                                    �J.�     !�             ���FHDB H�          ��;�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       `"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ���������      b�                           
                                                      (  �       ~�GBTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    OHDR                       ?      @ 4 4�     +         �                    �     �     N         |G            [                                                                                               ?p�FRHP               ��������r      �(                                                                                 (  �.       �q�BTHD 	      d(z=             �%`�          ;}��FHDB           �Uһ     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      BTHD       d(�j     
 
       �֦�P�FSSE (;     K �    ��FSSE �j     � M    u��          BTHD       d()             Ď�FRHP               ���������      �                           
                                                      (  t       غBiBTHD       d(@     
 
       �l�;BTHD 	      d(@     
 
       M��FSHD  �                             P x (        �x                   t[�]BTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    OCHK    1            �  	   0   REFERENCE_LIST 6     dataset        dimension                         ��            �            �q            !b            ��             �            �^�BTHD       d(?~     
 
       �֊BTHD 	      d(?�     
 
       )i\}FSSE x.     � M    ��By     �Q~OHDR4                                                 +         �                   PZ                      �O     ��     ��         ?      @ 4 4�   I                                                                             :�cFSSE ��     V �    ]{'� �FRHP               ��������K      (;                                                                                 (  |A       m��2            �3a�FHDB           f��     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                                       �"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FSHD  r                             P x (        �                   	�0�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OHDR`4    �                    �                       +         �                   @�     �                ��     �.     �         ?      @ 4 4�   �                                                                                                                                                       C\  y}
9FSHD  �                             P x (        n                   c�DPBTHD       d(��             �/���(  PmXhFHDB �
          ����     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       �"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FSHD  K                             P x (        �                   <i:BTLF  W    D     �$    1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� y   b     �]Y� G   2     f�� �    5     bS�� >   L  	   c���     A      c���    @     K�������                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    5         @      G   2      y   b      �   2         1      >   L  	    �   +  
   ���                                                                                                                                                                                                                                                                                                                                                                       OHDR�                      ?      @ 4 4�     +         �                   
 ]�     �     �         �r            F                                                                                                                                                                                                                                                                                                                                         Τq�       ��>FHDB �          �(�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools                            	   ��      �     ��      �       serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                           FHDB ��         O�$       one_way�	     %       cost_flow_in�     &       latitudez?     '       	longitude{     (       sink_use_max-     )       source_use_max�&     *       definition_matrix��     +       distance�     ,       timestep_resolution��     -       timestep_weights?�                                                                                                                                                                                        TREE  ����������������                       #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ��������V      ��                                                                                 (  PV       .��*BTLF  _    D     �$ �   1     +�
3 �    7     ��9    +  
   e>� �   2     Ŵ�� �   /     �]Y� O   2     f�� �    5     bS��    l  	   c���     I      c���    @     K�����!�                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    5         @      O   2      �   /      �   2      �   1         l  	       +  
   {�K�                                                                                                                                                                                                                                                                                                                                                                       OCHK    �     `       �    0   REFERENCE_LIST 6     dataset        dimension BTHD 	      d((�             8�pf         bFRHP               ��������J      �U                                                                                 (  "�       ���BTHD 	      d(��             �P�FSHD  J                             P x (        q�                   0]�GBTHD       d(��     
 
       ��zϊFSSE :�     J �    ,6 �BTHD 	      d(��     
 
       ķ           ȋ�FHDB �O          �.     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                       TREE  ����������������B                                       (#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTIN         |    4     �h     V��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF         ,  "         N  -         {  ,         �  '         �           �  )                    *           D           b            z   !        �   "        �   #        �   $           %        /   &        N   '        i   (        �   )        �  ! *        �  $ +        �   ,          & -        *  # 
�H�                                                                                                                                              FSHD  �                             P x (        �                   ��BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB ]�          ��f     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                       j#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                      ?      @ 4 4�     +         �                   ��               
 �}     "     H      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             n_�FRHP               ���������      x.                           
                                                      (  ��       
���BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    OHDR     �      �          ?      @ 4 4�     +         �                   
 ��     K�     ů              �      [                                                                                               ��q%BTHD       d(��     
 
       `�s#FSSE �U     J �    �N�� �   BTHD       d(��             t��    ZI��FHDB �}          ���     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                       �#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FSHD  �                             P x (        C�                   �q dimension           
       
       e            i0            w7            #�            f�            d�            v�            ��            �@FSHD  �                             P x (        ��                   ;W��FSSE �     � M    `�k�FSSE ��     r �    ���         FRHP               ��������J      :�                                                                                 (  @�       �ދ�BTHD 	      d(��             ���1FSHD  J                             P x (        �U                   ��j�BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       FSSE �1       �   �   �  M   u�5]FSSE s�     � M    v���ataset        dimenBTHD 	      d(��     
 
       ��Eh;�             FRHP               ��������r      ��                                                                                 (  z�       ���FSHD  r                             P x (        ^�                   r
��FRHP               ���������      �                           
                                                      (  ��       �BTHD       d(��     
 
       ��Z�    s��OHDRi4                                                            +   �    ��                     
 b�     �U     V                     �                                                                                                                                                                               �Y3{        j��FHDB ��          r�<�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  �����������������                                       �#             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       OCHK    0   REFERENCE_LIST 6     dataset        dimension                         ;�             e            i0            d�            v�            �@            [            �q             !b             ��            ,�            ��            ����FSSE �     � P    g�"BTHD 	      d(�      
 
       @�v� {           BTHD       d((�             N�{�FHDB �T          ��.G     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  �����������������                                       C$             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ���������      �                           
                                                      (  ��       T�BTLF  �        I�B i   1     +�
3 _    7     ��9 %   +  	   e>� 7   2     Ŵ��    /     �]Y� �    2     f�� �   l     c���     I      c��� �    @     K�����ߘ                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    7      �    @      �    2         /      7   2      i   1      �         �   l      %   +  	   �~$                                                                                                                                                                                                                                                                                                                                                                                    OCHK    �             �    0   REFERENCE_LIST 6     dataset        dimension                         ;�              e             i0             w7             Tk             d�             v�             ��             �             �@             o�             ��             ,�             �             z?             {             -            �&            ��             ~��OHDR     �      �          ?      @ 4 4�     +         �                   
 ��     ��     �         ��     �      {                                                                                                                                                                                                                                                                                                                                                                                              �3��  ��            ��C�FHDB b�          ��wB     _Netcdf4Coordinates                                   
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         dtype          bool     DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                 TREE  ����������������                                       �$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OCHK    �     `       ,     0   REFERENCE_LIST 6     dataset        dimension                         ;�             b�              e            i0            w7            Tk            #�             f�             d�            v�            ��            �            �@            [             �q            _�             o�            ��            ��             !�             �             ,�            %�             �	             �            -            �&            ��            �             �)
FHDB �          ��|     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������.                       %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    OCHK    �(     @          0   REFERENCE_LIST 6     dataset        dimension                         e            i0            w7            #�            f�            d�            v�            ��            �@            -             �&             ��             �O�FHDB ��          %$�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       0%             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  FRHP               ���������      s�                           
                                                      (         S�'KFSHD  �                             P x (        ��                   pt��BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    OCHK    �(     @      ,    0   REFERENCE_LIST 6     dataset        dimension                         e            i0            w7            #�            f�            d�            v�            ��            �@            -             �&             ��             ?�             �H_uFHDB ��          �l�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       S%             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             