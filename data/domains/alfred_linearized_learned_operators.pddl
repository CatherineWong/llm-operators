(define (domain alfred)
    (:requirements :adl
    )
    (:types
        agent location receptacle object rtype otype
    )
    (:constants
        CandleType - otype
        ShowerGlassType - otype
        CDType - otype
        TomatoType - otype
        MirrorType - otype
        ScrubBrushType - otype
        MugType - otype
        ToasterType - otype
        PaintingType - otype
        CellPhoneType - otype
        LadleType - otype
        BreadType - otype
        PotType - otype
        BookType - otype
        TennisRacketType - otype
        ButterKnifeType - otype
        ShowerDoorType - otype
        KeyChainType - otype
        BaseballBatType - otype
        EggType - otype
        PenType - otype
        ForkType - otype
        VaseType - otype
        ClothType - otype
        WindowType - otype
        PencilType - otype
        StatueType - otype
        LightSwitchType - otype
        WatchType - otype
        SpatulaType - otype
        PaperTowelRollType - otype
        FloorLampType - otype
        KettleType - otype
        SoapBottleType - otype
        BootsType - otype
        TowelType - otype
        PillowType - otype
        AlarmClockType - otype
        PotatoType - otype
        ChairType - otype
        PlungerType - otype
        SprayBottleType - otype
        HandTowelType - otype
        BathtubType - otype
        RemoteControlType - otype
        PepperShakerType - otype
        PlateType - otype
        BasketBallType - otype
        DeskLampType - otype
        FootstoolType - otype
        GlassbottleType - otype
        PaperTowelType - otype
        CreditCardType - otype
        PanType - otype
        ToiletPaperType - otype
        SaltShakerType - otype
        PosterType - otype
        ToiletPaperRollType - otype
        LettuceType - otype
        WineBottleType - otype
        KnifeType - otype
        LaundryHamperLidType - otype
        SpoonType - otype
        TissueBoxType - otype
        BowlType - otype
        BoxType - otype
        SoapBarType - otype
        HousePlantType - otype
        NewspaperType - otype
        CupType - otype
        DishSpongeType - otype
        LaptopType - otype
        TelevisionType - otype
        StoveKnobType - otype
        CurtainsType - otype
        BlindsType - otype
        TeddyBearType - otype
        AppleType - otype
        WateringCanType - otype
        SinkType - otype

        ArmChairType - rtype
        BedType - rtype
        BathtubBasinType - rtype
        DresserType - rtype
        SafeType - rtype
        DiningTableType - rtype
        SofaType - rtype
        HandTowelHolderType - rtype
        StoveBurnerType - rtype
        CartType - rtype
        DeskType - rtype
        CoffeeMachineType - rtype
        MicrowaveType - rtype
        ToiletType - rtype
        CounterTopType - rtype
        GarbageCanType - rtype
        CoffeeTableType - rtype
        CabinetType - rtype
        SinkBasinType - rtype
        OttomanType - rtype
        ToiletPaperHangerType - rtype
        TowelHolderType - rtype
        FridgeType - rtype
        DrawerType - rtype
        SideTableType - rtype
        ShelfType - rtype
        LaundryHamperType - rtype

    )
    ;; Predicates defined on this domain. Note the types for each predicate.
(:predicates
	(atLocation ?a - agent ?l - location) 
        (receptacleAtLocation ?r - receptacle ?l - location) 
        (objectAtLocation ?o - object ?l - location)
        (inReceptacle ?o - object ?r - receptacle) 
        (receptacleType ?r - receptacle ?t - rtype) 
        (objectType ?o - object ?t - otype) 
        (holds ?a - agent ?o - object) 
        (holdsAny ?a - agent) 
        (holdsAnyReceptacleObject ?a - agent) 
        
	    (openable ?r - receptacle) 
        (opened ?r - receptacle) 
        (isClean ?o - object) 
        (cleanable ?o - object) 
        (isHot ?o - object) 
        (heatable ?o - object) 
        (isCool ?o - object) 
        (coolable ?o - object) 
        (toggleable ?o - object)  
        (isToggled ?o - object) 
        (sliceable ?o - object) 
        (isSliced ?o - object) 
 )
(:action PickupObjectNotInReceptacle
        :parameters (?a - agent ?l - location ?o - object)
        :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (not (holdsAny ?a))
            (forall
                (?re - receptacle)
                (not (inReceptacle ?o ?re))
            )
        )
        :effect (and
            (not (objectAtLocation ?o ?l))
            (holds ?a ?o)
            (holdsAny ?a)
        )
    )

(:action PutObjectInReceptacle
        :parameters (?a - agent ?l - location ?ot - otype ?o - object ?r - receptacle)
        :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (objectType ?o ?ot)
            (holds ?a ?o)
            (not (holdsAnyReceptacleObject ?a))
        )
        :effect (and
            (inReceptacle ?o ?r)
            (not (holds ?a ?o))
            (not (holdsAny ?a))
            (objectAtLocation ?o ?l)
        )
    )
    
(:action PickupObjectInReceptacle
        :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
        :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (inReceptacle ?o ?r)
            (not (holdsAny ?a))
        )
        :effect (and
            (not (objectAtLocation ?o ?l))
            (not (inReceptacle ?o ?r))
            (holds ?a ?o)
            (holdsAny ?a)
        )
    )

    (:action RinseObject_2
        :parameters (?toolreceptacle - receptacle ?a - agent ?l - location ?o - object)

        :precondition (and 
		(receptacleType ?toolreceptacle SinkBasinType)
		(atLocation ?a ?l)
		(receptacleAtLocation ?toolreceptacle ?l)
		(objectAtLocation ?o ?l)
		(cleanable ?o)
		)
        :effect (and 
		(isClean ?o)
		)
)

(:action TurnOnObject_2
        :parameters (?a - agent ?l - location ?o - object)

        :precondition (and 
		(atLocation ?a ?l)
		(objectAtLocation ?o ?l)
		(toggleable ?o)
		)
        :effect (and 
		(isToggled ?o)
		)
)

(:action CoolObject_0
        :parameters (?toolreceptacle - receptacle ?a - agent ?l - location ?o - object)

        :precondition (and 
		(receptacleType ?toolreceptacle FridgeType)
		(atLocation ?a ?l)
		(receptacleAtLocation ?toolreceptacle ?l)
		(holds ?a ?o)
		)
        :effect (and 
		(isCool ?o)
		)
)
(:action SliceObject_1
        :parameters (?toolobject - object ?a - agent ?l - location ?o - object)

        :precondition (and 
		(objectType ?toolobject ButterKnifeType)
		(atLocation ?a ?l)
		(objectAtLocation ?o ?l)
		(sliceable ?o)
		(holds ?a ?toolobject)
		)
        :effect (and 
		(isSliced ?o)
		)
)
(:action SliceObject_0
        :parameters (?toolobject - object ?a - agent ?l - location ?o - object)

        :precondition (and 
		(objectType ?toolobject KnifeType)
		(atLocation ?a ?l)
		(objectAtLocation ?o ?l)
		(sliceable ?o)
		(holds ?a ?toolobject)
		)
        :effect (and 
		(isSliced ?o)
		)
)
(:action MicrowaveObject_0
        :parameters (?toolreceptacle - receptacle ?a - agent ?l - location ?o - object)

        :precondition (and 
		(receptacleType ?toolreceptacle MicrowaveType)
		(atLocation ?a ?l)
		(receptacleAtLocation ?toolreceptacle ?l)
		(holds ?a ?o)
		)
        :effect (and 
		(isHot ?o)
		)
)

)