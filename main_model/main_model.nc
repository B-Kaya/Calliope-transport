�HDF

         ���������     0       p�ғOHDRJ"     .       ��      Z-     �-     
          �      	      >	       �                                                                                                                                                                                                                                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                B!��FRHP              �      �0      d	       @              (             ��                                            (  ��       �%RoBTHD       d(�	              Io�BTHD 	      d(�              ���FSHD  Q	                            P x (        1�      D       D       rG�BTLF  �'   R  
   �d  {    <     ��          j�    O     >��  �&   @     �*% s   E     l3�- B   1     +�
3 H    3     ��!6    5    �Q@     y    kW�G K   (     Ce`     2      @�k �&   R     �s� �    (     ʻ#� P   R     kϬ� �&   $     ���    .     Ŵ�� �   2     �]Y� �   r     f�� E'   O  	   1��� �    "     ��N� s   @     K���+�%                                                                                                                                BTLF 	     2       H    3      {    <      �    (      �    "          y     �&   @      �&   $      �&   R      E'   O  	    �'   R  
       O      P   R         5               K   (      s   @      �   r      �   2         .      B   1      s   E     9d��                                                                                                                                                                                                                        BTHD       d(׷              ��*c     applied_math    ٩     history:
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
      where: techs=aluminium_transport_tech
      equations:
      - expression: flow_out[nodes=Iceland, carriers=aluminium] <= ship_depart * 20000
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
      - expression: "flow_out[nodes=Netherlands, carriers=aluminium]  == default_if_empty(\n\
          \     roll(flow_out[nodes=Iceland, carriers=aluminium], timesteps=72), \n\
          \     default=0\n   )"
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
BTLF �      ��             T�B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              FSSE ��     � 	    %�U�FRHP               ���������      �                           	                                                      (  ��       �]��BTHD       d(��     	 	       �6BTHD 	      d(��     	 	       @a�BTHD       d(��     
 
       ��]S             ��RBTHD 	      d(�             5�MFSHD  ~                             P x (        ��                   �S�BTHD 	      d(��     
 
       ���            5��JBTHD       d(Ǡ     
 
       M�-            gzo,BTHD 	      d(6�     
 
       �,] BTHD 	      d(Ǣ     
 
       I�X$FSSE �     � x    ����OHDR+                                     *       ׿      �       ��     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE #        NAME    	      carriers   ��F�OCHK   ^      b      $        NAME    
      timesteps +        _Netcdf4Dimid                :        units          hours since 2005-01-01 00:00:001    	    calendar          proleptic_gregorian  ��1pOCHK   &y     �       +        _Netcdf4Dimid                  %���          D,��FRHP               ���������       )�                                                                                  (  ��        υ�BTHD       d({�              �Q�BTHD 	      d({�              �C                  GCOL                        applied_math                  config                defaults
              �      p               t              p     �               �               �               �               �              aluminium_transport_tech�              fuel_supply     �              aluminium_supply�              aluminium_demand�               �              �     �              p     �               �               �              Netherlands     �              Iceland �               �              �     �              p     �              ]�                                               fuel          	       aluminium                    �                  p                  ]�                   r�                   �     !             p     "             ]�      #             r�     $             �     %             p     &             r�     '             �     (             ]�      )             r�     *             p     +             r�     ,             p     -             r�     .             �     /             p     0             ]�      1             r�     2             �     3             p     4             ]�      5             r�     ?              @             �     A             p     B             ��      C             r�     D             �     E             p     F             ��      G              H             monetaryO             �     P             p     Q             ]�      R             r�     T             p     U             ]�      Y             ]�      Z             p     [             ��      ]             ]�      ^             ��      _             ��      `              a             p     b              c              d              e              f             transmission    g             supply  h             supply  i             demand  j              l             �     m             p     n              o              p              q              r              s              t              u              v      	       aluminium       w              x              y              z              {             fuel    |      	       aluminium       }              �             �     �             p     �             ]�      �              �             p     �              �              �              �              �             #8465A9 �             #E37A72 �             #F9CF22 �             #072486 �             p     �              �             p     �              �              �              �              �             Aluminium Transport     �             Fuel Supply     �             Aluminium Supply�             Aluminium Demand�             �     �             p     �             ]�      �             p     �             �     �             �     �             ��      �             p     �             r�     �             �     �             p     �             r�     �             �     �             p     �             �     �             p     �             ]�      �             p     �             r�     �             r�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDR(                                     *       ׿      ?      ��      Q           
 ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          costs   +���           9OHDR4                                                  ?      @ 4 4�     +         �                   �                       �      y�      ��       5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        9YFSHD  �                              P x (        t�                    %m�BTLF  _    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    D     Q�ń T   +     e>� �   2  
   Ŵ�� S   2  	   �]Y� !   2     f�� V   5     bS�� �   l     c���     I      c��� �    o     G��� �   @     K���r                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    D      �    o      V   5      �         �   7      �   @      !   2      S   2  	    �   2  
    �   1      �   l      T   +     �A3�                                                                                                                                                                                                                                                                                                                                FSSE �     �  m    '��%FSSE �     �  i    �O��SSE Q     �  C    ��@�BTHD       d(ei             �}ΙBTHD       d(,W     
 
       *   �  "    /��BTHD       d(�             aC{BBTHD 	      d(�             /$2�FSSE )�      �      ϻ��FRHP               ���������      �-                          )                                                         ��      �%�           ��-FSSE d	        �'      �    H �    �              ��RFSSE ��      �  M    F��FHDB �            �5c     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title    )      Technology flow (a.k.a. nominal) capacity     description    N      A technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �      ׿      �      ׿      �        _Netcdf4Dimid                                                                                                                                                  FHDB �          o5�     config         init:
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
  ensure_feasibility: true
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
      model.yaml     serialised_single_element_list  ?      @ 4 4�         serialised_bools  ?      @ 4 4�                               FHDB �           ��� 	    defaults    [     bigM: 1000000000.0
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
      main model     timestamp_model_creation  ?      @ 4 4�                �e��e�A     timestamp_build_start  ?      @ 4 4�                6>�e�A     timestamp_build_complete  ?      @ 4 4�                P���e�A                          FHDB �           �8�R     termination_condition          optimal     calliope_version_defined          0.7.0     calliope_version_initialised    
      0.7.0.dev6     applied_overrides            	    scenario          None     timestamp_solve_start  ?      @ 4 4�                ����e�A     timestamp_solve_complete  ?      @ 4 4�                `#�e�A     serialised_dicts                               ׿            ׿            ׿              serialised_nones          scenario     serialised_sets  ?      @ 4 4�         _NCProperties    "      version=2,netcdf=4.9.2,hdf5=1.14.4                                                                                                                                                                                                                                                                                                                                        FHIB �           ��      ��      ��������������������������������������������������      ������������������������pf�TREE  ����������������3                                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         OHDR�                      ?      @ 4 4�     +         �                                   �      ��      ��       #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      �,8FRHP               ���������       ��                                                                                  (         �c�BTHD       d(J�              ��uBTHD 	      d(J             � xqFSHD  �                              P x (        u�                    �f��BTLF  W    D     �$ �   1     +�
3 y        �I75 �   7     ��9 �    -     Q�ń �   +     e>� s   2  
   Ŵ�� A   2  	   �]Y�    2     f�� D   5     bS��    L     c���     A      c��� �    |     G��� �   @     K���Җ�]                                                                                                                                                                                                                                                                        BTLF 	     A       W    D      �    -      �    |      D   5      y         �   7      �   @         2      A   2  	    s   2  
    �   1      �   +         L     o �                                                                                                                                                                                                                                                                                                                                FRHP               ��������|       �                                                                                 (  �n       :G��           �$�x^c`����:�����b`������?~0������; a=�c=�U_ ��PFHDB �           �b�5     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      �     title          Link flow capacity     description    [      A transmission technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         _Netcdf4Dimid                     DIMENSION_LIST                              ׿      t                                                                                                                                                                                      TREE  ����������������                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR]D                         �                           �          ?      @ 4 4�     +         �                                 �          �     (�      N�       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               Ť�	FRHP               ��������*       J                                                                                 (         8=u�FSHD  *                              P x (        �1                   =�r�BTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    *     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� h   2     f�� �   5     bS�� /   |     c���     M      c��� �    �     G��� (   @     K���>���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    *      �    �      �   5      �          �   7      (   @      h   2      �   2  	    �   2  
    �   1      /   |      �   +     x��                                                                                                                                                                                                                                                                                                                                BTHD       d(�     
 
       ���BTHD 	      d(��     
 
       ��FSHD  �                             P x (        �                   ��j�BTHD       d(m�             >�U4         ��C�BTHD 	      d(ek             h���FSHD  |                              P x (        m�                   =�=      ����x^c`�:~�����ޡ ":FHDB �          ��S     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier outflow     description    �      The outflow of a technology per timestep, also known as the flow discharged (from `storage` technologies) or the flow received (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿           ׿           ׿           ׿             _Netcdf4Dimid                                                           TREE  �����������������                                               �9                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR D                         �                           �          ?      @ 4 4�     +         �                   h@              �          x4     
5     05                                         ����BTHD       d(�-     ) )       ��v6BTHD      d(_a      )       ]�7FSHD  �                              P x          tk     ,       ,       �#��BTLF �?� >   �g {  ! i�8 �   y}W !   ���    # �rQ' �    5�. =   �8 �    ���I �  ) )m�M �  & y��P |    `��P    ��-S �  , w�iV �   ���X 0   ���[ �  # �^P^    _.]c ]   ��l �     'q l  " ��{� u   q��� !   1M7� ?  " D9� �   �ԕ� �   ��_� `    J鱷 �  ' 7��� a  - ĕ�� �    .BD� L    ho�     ��� �   S�:� W   Ѧ� -    ���   * d�� E    �;�� �  " yܴ� �  ! �}"� �   XX� �  $ ��� �   Ds]{                                                   BTLF         �            �             �            �    	           
        !           >           ]           {  !         �  #         �  "           *         �           -                        E    x���                                                                                                                                                                                                                                                                      FSSE J     *  �    ��FRHP               ��������:      �                                                                                 (  ��       8�
� �|            �;؋FRHP               ��������'      �                                                                                 (  ��       �Ȧ�           "ֻOHDR 4                  �                    �         +         �                   Z           �          3P     �P     �P         ?      @ 4 4�   3                                                       �f{�                FRHP               ��������4       V5                                                                                 (  h:       �2�BTHD       d(�5             ���lBTHD 	      d(�7             ��FSHD  4                              P x (        P                   ��'!BTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    )     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� ^   2     f�� �   5     bS�� %   |     c���     M      c��� �    �     G���    @     K���	;Lb                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    )      �    �      �   5      �          �   7         @      ^   2      �   2  	    �   2  
    �   1      %   |      �   +     8�                                                                                                                                                                                                                                                                                                                                x^�ڡ�0 CAFc3,{��l����U^�                                                          `�6��C��e�     rg�F�= ��^ *<���e��(�`M�3B��  'wFh�� ����	��Qv��{ ֔?]h��  �����{ �Y}!K��FHDB x4          �ThE     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow     description    �      The inflow to a technology per timestep, also known as the flow consumed (by `storage` technologies) or the flow sent (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿            ׿      !     ׿      "     ׿      #       _Netcdf4Dimid                                                                     FHDB ��          �|8�       nodes�            techsp            carriers]�             	timestepsr�            costs��             flow_cap�             link_flow_capF�             flow_outO            flow_inN,     	       
source_use\3     
       
source_capVf            unmet_demand�|            ship_depart��            ship_available�            flow_out_inc_eff��            flow_in_inc_eff��            cost�             TREE  �����������������                                               cU                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������       �                                                                                 (  g       F��BTHD       d(p�             ���}      ��            � �{FRHP               ���������      �                           
                                                      (  ��       8]�FSSE V5     4  �    ��4    FRHP               ���������       Q                                                                                 (  V       '�NCBTHD       d(cQ             VMa�BTHD 	      d(cS             ��FSHD  �                              P x (        @�                   ����BTLF  _    D     �$ �   1     +�
3 y         �I75 �   7     ��9 �    *     Q�ń C   +     e>� t   2  
   Ŵ�� B   2  	   �]Y�    2     f�� D   5     bS�� �   l     c���     I      c��� �    w     G��� �   @     K����X                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    *      �    w      D   5      y          �   7      �   @         2      B   2  	    t   2  
    �   1      �   l      C   +     O��                                                                                                                                                                                                                                                                                                                                x^�ڡ� A*��X0�����3؃��a�.                                                   ��d�       �s�0@v ���{    �;?  ��d�  �V � h���Qv �j  jl0@v ��d�  ��Ӆ7�� Z�_d6�FHDB 3P          爵�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Source flow use     description    V      The carrier flow consumed from outside the system boundaries by a `supply` technology.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      $     ׿      %     ׿      &       _Netcdf4Dimid                                                                                                                                                                   TREE  ����������������p                                       Gn                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR$                                    ?      @ 4 4�     +         �                   �r                   J     ��      n      J                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             c"0�BTLF  [    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    /     Q�ń Y   +     e>� �   2  
   Ŵ�� h   2  	   �]Y� 6   2     f�� k   5     bS�� �   \     c���     E      c��� �    �     G��� �   @     K����闯                                                                                                                                                                                                                                                                        BTLF 	     E       [    D      �    /      �    �      k   5      �         �   7      �   @      6   2      h   2  	    �   2  
    �   1      �   \      Y   +     ���                                                                                                                                                                                                                                                                                                                                FRHP               ���������       ��                                                                                 (  ��       �՜BTHD       d(�             õ7�BTHD 	      d(�             �2�v�x^�ء�@C�+�ΰ�ESH|$���٭�L�                            ��.(��   ��n(���D�c`��  ����a��a��  ���6�FHDB J          �|�+     _Netcdf4Coordinates                                    _FillValue  ?      @ 4 4�                      �     title          Source flow capacity     description    |      The upper limit on a flow that can be consumed from outside the system boundaries by a `supply` technology in each timestep.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �      ׿      �        _Netcdf4Dimid                                                                                                                                             TREE  ����������������                               ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�4                  �                    �          ?      @ 4 4�     +         �                   ��           �         
 �O     �     �      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        R��_BTLF  _    D     �$ �   1     +�
3 �    7     ��9 M   +  	   e>� ~   2     Ŵ�� L   2     �]Y�    2     f�� �   l     c���     I      c��� �    @     K�����1�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   2      ~   2      �   1      �   l      M   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    FSHD  �                              P x (         �                   �Ⱦ\FSSE �     � x    "�f$BTHD 	      d(p�             ���^FSSE �     : �    	W/$FSSE  �     �  R    �nM    Xi3             ��Ɔx^c` t@��0� �z�z�z   kM
VFHDB �O          yژ�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      '     ׿      (     ׿      )       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                         TREE  ����������������L                                       _�                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR�$           �             �          ?      @ 4 4�     +         �                   ��        �          �2     :     ��      5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        u�63BTHD 	      d(m�             J���FSHD  '                             P x (        ;�                   Is�kBTLF  [    D     �$ !   1     +�
3    7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� �   2     f�� R   \  	   c���     E      c��� �    u     G��� K   @     K���ٱ�                                                                                                                                                                                                                                                                                                                           BTLF 	     E       [    D      �    u         7      K   @      �   2      �   2      �   2      !   1      R   \  	    �   +  
   +�m�                                                                                                                                                                                                                                                                                                                                                                       FSSE �     |  �    �FRHP               ���������        �                                                                                 (  ��       �*�MBTHD       d(r�             [�
�FSSE Q     �  n    y�J��X�x^��1  AQ�c+ئ���Ѱ��            ��� ^5 @� �UP���&��y�2�FHDB �2          �t�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      �     description    T      1 if a ship departs with an aluminium shipment from Iceland in this timestep, else 0 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      *     ׿      +       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                       TREE  ����������������A                               j�                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�$           �             �          ?      @ 4 4�     +         �                   ��        �          2     ʮ     �      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     �e�$BTHD       d(h�             �i��BTHD 	      d(h�             �^Q�FSHD  :                             P x (        P�                   � �dBTLF  [    D     �$    1     +�
3    7     ��9 �   +  
   e>� �   2     Ŵ�� �   2     �]Y� x   2     f�� ?   \  	   c���     E      c��� �    b     G��� 8   @     K����t�                                                                                                                                                                                                                                                                                                                           BTLF 	     E       [    D      �    b         7      8   @      x   2      �   2      �   2         1      ?   \  	    �   +  
   ���                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������       �                                                                                 (  �!       Ռ5kBTHD 	      d(�             �N1FSSE ��     �  P    r%��FSSE �     ' �    �yǵ��            ����x^�ҡ  �}�i��D�'�]rC�                 �q�p`��m ",���D�t7FHDB 2          �ʌ�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      �     description    A      Number of ships available at origin (Iceland) at a given timestep 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      ,     ׿      -       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                          TREE  ����������������A                               v�                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDRD                         �                           �          ?      @ 4 4�     +         �                   ��              �          ��     �     ��      *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ��aBTHD 	      d(r�             ��FSHD  �                              P x (        k�                   H�N3BTLF  c    D     �$ z   1  
   +�
3 m   7     ��9 �    ;     Q�ń '   +     e>� H   2  	   Ŵ��    2     �]Y� �   2     f�� 8   5     bS�� �   |     c���     M      c��� �    V     G��� �   @     K���!���                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    ;      �    V      8   5      m   7      �   @      �   2         2      H   2  	    z   1  
    �   |      '   +     5���                                                                                                                                                                                                                                                                                                                                             OHDR     �      �                @    *         �    ��      �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE  
              옫  ��уOCHK    @2             +        _Netcdf4Dimid                   ��     :       v                                                                                                                          ���          :�x^�ҡ  �}�i��D�'�]rC�                 �q�p`��m ",���D�t7FHDB ��          >��     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title           Carrier outflow including losses     description    5      Outflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      .     ׿      /     ׿      0     ׿      1       _Netcdf4Dimid                                                                                                                                                                                               TREE  �����������������                                               �                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR	D                         �                           �          ?      @ 4 4�     +         �                   ��              �          em     �m     n      (                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ��"BTLF  c    D     �$ x   1  
   +�
3 k   7     ��9 �    :     Q�ń %   +     e>� F   2  	   Ŵ��    2     �]Y� �   2     f�� 6   5     bS�� �   |     c���     M      c��� �    U     G��� �   @     K�����o�                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    :      �    U      6   5      k   7      �   @      �   2         2      F   2  	    x   1  
    �   |      %   +     �5��                                                                                                                                                                                                                                                                                                                                             OCHK   �     �               NAME          nodes +        _Netcdf4Dimid                   YobqFSSE �V     t �    2�X]FHIB ��          h>     g     /W     ��������Z�|�BTHD       d(�             ���� ��             ��x^�ڡ�0 CAFc3,{��l����U^�                                                          `�6��C��e�     rg�F�= ��^ *<���e��(�`M�3B��  'wFh�� ����	��Qv��{ ֔?]h��  �����{ �Y}!K��FHDB em          tȾ     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow including losses     description    4      Inflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      2     ׿      3     ׿      4     ׿      5       _Netcdf4Dimid                                                                                                                                                                                                 TREE  �����������������                                               �                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�D                         �                           �          ?      @ 4 4�     +         �                   g
              �          �N     BO     *�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ꘵�FSHD  �                              P x (        {�                    �Έ�BTLF  c    D     �$ �   1     +�
3 a   '     �I75 �   7     ��9 �    3     Q�ń B   +     e>� c   2  
   Ŵ�� 1   2  	   �]Y� �   2     f�� ,   5     bS�� �   |     c���     M      c��� �    R     G��� �   @     K���/���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    3      �    R      ,   5      a   '      �   7      �   @      �   2      1   2  	    c   2  
    �   1      �   |      B   +     ]Ě                                                                                                                                                                                                                                                                                                                                OHDR(                                     *       ׿      p       T     Q             ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          techs   u�UFSSE u�     � d    �H:~FRHP               ��������~      /�                                                                                  (  �       �_[�L�|OCHK   ��     B      +        _Netcdf4Dimid                  ���          �>�x^�ڡ� A*��X0�����3؃��a�.                                                   ��d�       �s�0@v ���{    �;?  ��d�  �V � h���Qv �j  jl0@v ��d�  ��Ӆ7�� Z�_d6�FHDB �N          	p+     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Variable operating costs     description    1      The operating costs per timestep of a technology.     default                                 unit          cost_per_time 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      @     ׿      A     ׿      B     ׿      C       _Netcdf4Dimid                                                                                                                                                                    FHDB ��         y�       cost_operation_variable�            capacity_factor<S            systemwide_capacity_factorNn            systemwide_levelised_cost�            total_levelised_cost|u            bigM	�            objective_cost_weightsw�            	base_techF�            carrierL�            carrier_outp�             color��     !       flow_cap_max4�     "       name�     #       
carrier_inD�     $       flow_out_eff�      TREE  ����������������i                                               !                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�4                                                  ?      @ 4 4�     +         �                   �%                      h�     ��     ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ̤��FSHD  �                              P x (        ��                    z�fbBTLF  _    D     �$ �   1     +�
3 v        �I75 �   7     ��9 �    &     Q�ń >   +     e>� o   2  
   Ŵ�� =   2  	   �]Y�    2     f�� A   5     bS�� �   l     c���     I      c��� �    x     G��� �   @     K���k�                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    &      �    x      A   5      v         �   7      �   @         2      =   2  	    o   2  
    �   1      �   l      >   +     su�                                                                                                                                                                                                                                                                                                                                OHDR                                     *       ׿      �       �     w            ������������������������A         _Netcdf4Coordinates                            +        CLASS          DIMENSION_SCALE   ȍA�             �.�Sx^�ر  AF�e*֦��  %w3|ay                                  x�@��1PQv  ����	d�PQ��PQv  �s���FHDB h�          '!"�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Total costs     description    W      The total annualised costs of a technology, including installation and operation costs.     default                                 unit          cost 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      D     ׿      E     ׿      F       _Netcdf4Dimid                                                                                                                                                                        TREE  ����������������                                       �1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         x^c` hA��~0� �z�z�z   d�
   ׿      �      ׿      �      ׿      �      ׿      �      ׿      �      ׿      �   	   ׿           ׿           ׿      H  x^��1�0A3��qeɺ���ծ                                                                      ��`���0@v    ���e�   p�d��(��F�=    �N~Fh�� �����Px^c` |`x �큐� VOx^c�f� (�J�C�P�
B�(����`  �	�x^�faX���𐡊��}= #_
    ��.Ax^c` �� '0   ׿      i     ׿      h     ׿      f     ׿      g      ׿      }  	   ׿      |      ׿      z     ׿      {  	   ׿      v      ׿      w      ׿      x      ׿      y  x^c`�-��������B� ��Q   ׿      �     ׿      �     ׿      �     ׿      �  x^c`�:~|����ޡ !��   ׿      �     ׿      �     ׿      �     ׿      �  x^c`�-��p=�#a ���x^c`�~���޾ ��x^{i��h��Ѳ��C?��� (y`x^[�p�����d�3O�
p  G-x^c�f�f�u@�������Gɏ)���� ��
3x^�ű  ��H#Q����p/���$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I��5�$I�$I�$�8H�$I�$I���ܕ3�$I�$I�$��]=#I�$I�$I�����@x^�š   �N�,����]=2�"�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�$I�����$I�$I�$���I�$I�$I�dq��F�$I�$I�$���7�$I�$I�$�<R���x^c``BF �  ? x^c`x���`���x!���?����w� k4	x^��1  �5�#X�o@   ���JOp�x^��1  �5�#X�o@   ���JOp�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               OHDR4    �                    �                       +         �                   K�     �                q     �l     �         ?      @ 4 4�   I                                                                             �ύ7FRHP               ���������      *q                           
                                                      (  �v       ��sBTHD       d(|q     
 
       2�j���                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      OHDR�D                         �                           �          ?      @ 4 4�     +         �                   `              �         
 "V     ��      �V      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            �MDFRHP               ��������t      �V                           
                                                      (  \       ��
BTHD 	      d(,Y     
 
       �P�bFSHD  t                             P x (        ��                   .�%BTLF  c    D     �$ �   1     +�
3 �    7     ��9 a   +  	   e>� �   2     Ŵ�� P   2     �]Y�    2     f�� �   |     c���     M      c��� �    @     K������                                                                                                                                                                                                                                                                                                                                            BTLF 	     M       c    D      �    7      �    @         2      P   2      �   2      �   1      �   |      a   +  	   #��                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������      �                           
                                                      (  ��       L��BTHD       d(6�     
 
       qb~yFSSE *q     � d    <3�e   �9� FHDB "V          �`:"     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      O     ׿      P     ׿      Q     ׿      R       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                     TREE  �����������������                                               p2                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�$                                    ?      @ 4 4�     +         �                   �z                  
 B;     �;     q      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ^T��BTHD 	      d(|s     
 
       p�RFSHD  �                             P x (        �[                   x�+�BTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y�    2     f�� �   \     c���     E      c��� �    @     K���ۿ��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    @         2      H   2      z   2      �   1      �   \      9   +  	   �2k|                                                                                                                                                                                                                                                                                                                                                                                    OHDRH$                                   +         �                   ѩ                  
 �     ��      �          ?      @ 4 4�   �                                                                                                                                                   �(ڠ          m�ČFHDB B;          ^���     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      T     ׿      U       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                               3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�4                                                  ?      @ 4 4�     +         �                   ��                     
 ,[     �[     �       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        6�dFSHD  �                             P x (        ?�                    ��dBTLF  _    D     �$ �   1     +�
3 �    7     ��9 M   +  	   e>� ~   2     Ŵ�� L   2     �]Y�    2     f�� �   l     c���     I      c��� �    @     K�����1�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   2      ~   2      �   1      �   l      M   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������      ��                           	                                                      (  ��       ���BTHD       d(L�     	 	       
j�      ���BTHD       d(�             ۹�FSHD  �                             P x (        G�                   �u��OHDR4                                                 +         �                   ��                     
 ��     к      ��          ?      @ 4 4�   I                                                                             N�G          OQȊFHDB ,[          $�     _Netcdf4Coordinates                                      _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      Y     ׿      Z     ׿      [       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                        TREE  ����������������#                                       $3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         BTHD 	      d(K�             �rN�  ����FSSE �z     j �    ��4NBTHD       d(��             �\�  ��RFSSE �     J �    s):� BTHD 	      d(5�             �Ϻ�      w�             �S��FSSE /�      ~ �    ��G/                    FRHP               ���������      u�                           
                                                      (  ѥ       P�fFSHD  �                             P x (        �                   X�UBTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y�    2     f�� �   \     c���     E      c��� �    @     K���ۿ��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    @         2      H   2      z   2      �   1      �   \      9   +  	   �2k|                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������      �                           
                                                      (  �        v��BTHD 	      d(D�     
 
       �A��FSHD  �                             P x (        ��                   ��W&FHDB �          qO     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      ]     ׿      ^       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                               G3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDRb           ?      @ 4 4�     *         �    _3                 ������������������������D         _FillValue  ?      @ 4 4�                      �7    
    is_result                            A        default  ?      @ 4 4�                    e��A@        serialised_single_element_list  ?      @ 4 4�    2        serialised_dicts  ?      @ 4 4�    /        serialised_bools    	      is_result2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    0���OHDR�                      ?      @ 4 4�     +         �                   �                �     ��     	�       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 �� SBTLF  W    D     �$ �   1     +�
3 �    7     ��9 W   +  
   e>� �   2     Ŵ�� y   /     �]Y� G   2     f�� �    5     bS��    L  	   c���     A      c���    @     K�����                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    5         @      G   2      y   /      �   2      �   1         L  	    W   +  
   &��                                                                                                                                                                                                                                                                                                                                                                       FHDB �          ���     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      _       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                              TREE  ����������������                       g3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                                     *       ׿      `      t3     @           	 6�     Ȍ     ��      �                                                                    7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    L                                                                               O%��BTHD 	      d(L�     	 	       �և�FSHD  �                             P x (        ׹                    ���<BTLF  a   1     +�
3 W    7     ��9 �   +     e>� /   2     Ŵ��     /     �]Y� �    2     f�� �   L     c���     A      c��� �    @     K����;��                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    7      �    @      �    2          /      /   2      a   1      �   L      �   +     ��|                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                                   *       ׿      j      �3     �           	 �      ��      ��                                                                               7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    \                                                                                               �¹FSSE �     � J    �1�}FSSE /�     � u    �XM}OHDR0                      ?      @ 4 4�     +         �                    �     p�              R            �                                                                                                                                               ��T�BTHD       d(�             H�iOCHK,    0   REFERENCE_LIST 6     dataset        dimension                         �             O            N,            �|            ��            ��            <S            Nn            �             |u             p�            D�            ��            �vyŔ�FHDB 6�          ��|     _Netcdf4Coordinates                            
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      a       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       BTLF  e   1     +�
3 [    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   \     c���     E      c��� �    @     K���e���                                                                                                                                                                                                                                                                                                                                                             BTLF 	     E       [    7      �    @      �    2         /      3   2      e   1      �   \      �   +     3it.                                                                                                                                                                                                                                                                                                                                                                                                 OHDRH$                                   +         �                   �                   3z     @f     _e         ?      @ 4 4�   �                                                                                                                                                   ^��FSSE "g     � M    �@�Q�g��FSSE �     �     �(!BTHD       d(D�     
 
       �,4�                     FRHP               ���������      /�                           
                                                      (  ��       ���CFSHD  �                             P x (        �                   �r�QBTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    FHDB ��          [�      _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �     ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            FHDB �           I��`     _Netcdf4Coordinates                                
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      l     ׿      m       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                                       44                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         OHDRw                                     *       ׿      �      O4     @           
 Ǥ     b�     Y�      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          �~�8BTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    OHDR4                                                 +         �                   W/                     
 �$     $     �          ?      @ 4 4�   I                                                                             �FSSE      � J    *�C� <S BTHD 	      d(f     
 
       �T�FSSE >     r �    �=m�BTHD 	      d(u'     
 
       R���     ��)6FHDB Ǥ          md �     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDRU                                     *       ׿      �      
 �     W     �          �4     @        �                                                                                                                                                                                                         /-��BTHD       d(f     
 
       �Xp         FRHP               ��������r      >                                                                                 (  R       G[�BTHD 	      d(�	             �L�FSHD  r                             P x (        �                    ���sBTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OHDRm                      ?      @ 4 4�     +         �                    �;     8$     ^$         F            �                                                                                                                                                                                                            ���FSSE �     r �    5/�VBTHD       d(u%     
 
       �3C    ��cFHDB �          C���     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       �4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ���������                                 
                                                      (  8        [�OFSHD  �                             P x (        f                    9r-BTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    FSSE #%     � u    �a<�OHDR                       ?      @ 4 4�     +         �                   /Y               
 KN     �            [                                                                                               �T5�BTHD       d(/O     
 
       	�8 Nn  FSHD  r                             P x (        	                   �u�BTHD 	      d(/Q     
 
       z���   �;�FHDB �          �	�     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      BTHD       d()<             ��VdBTHD 	      d()>             f�Y�             FRHP               ���������      #%                           
                                                      (  W+       o>lFSHD  �                             P x (        f                   ؼ:/BTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    BTHD       d(tg     
 
       +U�  ��?�OHDR�                      ?      @ 4 4�     +         �                   
 jf     u)     �f         �p            �                                                                                                                                                                                                                                                                 o��F    A'�FHDB �$          r3�Q     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �     ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                                       �4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FRHP               ��������r      �                                                                                 (  B       �5�DBTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OHDR�4    �                    �                       +         �                   �     �                Q�     z     t�         ?      @ 4 4�                                                                                                                                                                                                                                                                                             /)����FHDB �;          �Q     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       �4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ���������      �N                           
                                                      (  /S       �w��FSHD  �                             P x (        �e                   ��d�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB KN          Z     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    FHDB ��         ��s�%       latitude�     &       	longitude�)     '       cost_flow_out��     (       sink_use_equals)@     )       source_use_max :     *       definition_matrix��     +       distanceg�     ,       timestep_resolution5�     -       timestep_weights��                                                                                                                                                                                                              TREE  ����������������                       5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTIN         `    �/     _c     �R#�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF         |            ?  "         a  -         �  ,         �  '         �           �  )         !           =           W            u   !        �   "        �   #        �   $        �   %           &        0   '        L    (        l  " )        �  ! *        �  $ +        �   ,        �  & -          # #`�C                                                                                                                                              BTHD 	      d(}             ��  NFSSE �N     � M    ��"FRHP               ���������      ]�                           
                                                      (  ��       3"        BTHD       d({             G��    FRHP               ���������      "g                           
                                                      (  �l       �%�BTHD 	      d(ti     
 
       �3!�FSHD  �                             P x (        &�                   ��	BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FSSE �-       �     �   �  7   C}]�FSSE �     � M    �O3�et        dimension                         O            N,            \3            �|            ��            �            ��            ��            �            <S            )@              :             5�             �t��       BTHD       d(K�             �Cɖ��FHDB jf          n!;G     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                       '5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OCHK    `2            �  	   0   REFERENCE_LIST 6     dataset        dimension                         �            �            �            |u            w�             ��             �{�2BTHD       d(5�             �c�     FRHP               ��������j      �z                                                                                 (  �       ��dFSHD  j                             P x (        �                   �<�BTLF  [    D     �$ �   1     +�
3 �    7     ��9 k   +  
   e>� �   2     Ŵ�� }   /     �]Y� K   2     f�� �    5     bS��    \  	   c���     E      c���    @     K�����r                                                                                                                                                                                                                                                                                                                           BTLF 	     E       [    D      �    7      �    5         @      K   2      }   /      �   2      �   1         \  	    k   +  
   ����                                                                                                                                                                                                                                                                                                                                                                       FSSE �     J �    E��BTHD       d(�     
 
       H�\ dataset        dimensiFRHP               ��������J      �                                                                                 (  K�       �ȁFSHD  J                             P x (        U�                   �{��BTHD       d(^�     
 
       �0�  ��7FSSE ��     r �    R�JEBTHD 	      d(^�     
 
       RC�VFSSE ]�     � M    ���W        DBTHD 	      d(��             ��E        �7TIFHDB 3z          T�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������"                               @5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          FRHP               ��������J      �                                                                                 (  �       �M�.FSHD  J                             P x (                           �BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       OHDR     �      �          ?      @ 4 4�     +         �                   
 �e     ��     ҽ         ��     �      {                                                                                                                                                                                                                                                                                                                                                                                              �y}W  �ТOHDRm                      ?      @ 4 4�     +         �                    �     )�     �         -�            �                                                                                                                                                                                                           �b��      :            ����OHDR 4                                                            +   �    ��                     
 �     2     ��                     B                                                                      ��1  ��eFHDB Q�          '     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �     ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������r                                       b5             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       FHDB q          ��@U     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �     ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������r                                       �5             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTHD       d(��     
 
       ��z�  �TFSSE þ     � P    /Ŀ�BTHD 	      d(��     
 
       ��F,                   FRHP               ���������      þ                           
                                                      (  ��       ���\BTHD 	      d(�     
 
       �7FSHD  �                             P x (        ��                   ���BTLF  �        I�B i   1     +�
3 _    7     ��9 %   +  	   e>� 7   2     Ŵ��    /     �]Y� �    2     f�� �   l     c���     I      c��� �    @     K�����ߘ                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    7      �    @      �    2         /      7   2      i   1      �         �   l      %   +  	   �~$                                                                                                                                                                                                                                                                                                                                                                                    OCHK     2             �    0   REFERENCE_LIST 6     dataset        dimension                         �              O             N,             \3             Vf             �|             ��             ��             �             �             <S             L�             p�             D�             �             �)             )@             :            ��             ��.FHDB �          s�H�     _Netcdf4Coordinates                                   
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         dtype          bool     DIMENSION_LIST                              ׿      �     ׿      �     ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                 TREE  ����������������                                       F6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FRHP               ��������r      ��                                                                                 (  -�       � ȍFSHD  r                             P x (        ��                   ^��BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OCHK    �1     @           0   REFERENCE_LIST 6     dataset        dimension                         �             F�              O            N,            \3            Vf            ��             �             ��            ��            �            �            <S            Nn             �            F�             L�            p�            ��             4�             �             D�            �             ��            )@             :            ��            g�             � g�FHDB �          &��3     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������#                       X6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FSHD  �                             P x (                           Q���BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB �e          ���x     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       {6             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR�    �      �          ?      @ 4 4�     +         �                   �     �         
 z�     U�     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             *w�FRHP               ���������      �                           
                                                      (  �       ]�&TFSHD  �                             P x (        �k                   g�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    OCHK    �;     @      <    0   REFERENCE_LIST 6     dataset        dimension                         O            N,            \3            �|            ��            �            ��            ��            �            <S            )@              :             5�             ��             "�4aFHDB z�          �C�     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ׿      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       �6             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             