;; ALFRED domain ground truth PDDL file | Author: zyzzyva@mit.edu
;; This edited form of the PDDL operator file was derived based on a combination of:
;; The ALFRED PutTaskExtended at: https://github.com/askforalfred/alfred/blob/master/gen/planner/domains/PutTaskExtended_domain.pddl
;; The alfworld version at: https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl
;; Changes to the original: removed costs, and removed foralls, which we don't check.
;; Removed is on check from toggling.
;; Flattened away the 'OR' to require the microwave, fridge, and sink.

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
    (:predicates
        (atLocation ?a - agent ?l - location) ; true if the agent is at the location
        (receptacleAtLocation ?r - receptacle ?l - location) ; true if the receptacle is at the location (constant)
        (objectAtLocation ?o - object ?l - location) ; true if the object is at the location
        (openable ?r - receptacle) ; true if a receptacle is openable
        (opened ?r - receptacle) ; true if a receptacle is opened
        (inReceptacle ?o - object ?r - receptacle) ; object ?o is in receptacle ?r
        (isReceptacleObject ?o - object) ; true if the object can have things put inside it
        (inReceptacleObject ?innerObject - object ?outerObject - object) ; object ?innerObject is inside object ?outerObject
        (receptacleType ?r - receptacle ?t - rtype) ; the type of receptacle (Cabinet vs Cabinet|01|2...)
        (objectType ?o - object ?t - otype) ; the type of object (Apple vs Apple|01|2...)
        (holds ?a - agent ?o - object) ; object ?o is held by agent ?a
        (holdsAny ?a - agent) ; agent ?a holds an object
        (holdsAnyReceptacleObject ?a - agent) ; agent ?a holds a receptacle object
        ;(full ?r - receptacle)                                    ; true if the receptacle has no remaining space
        (isClean ?o - object) ; true if the object has been cleaned in sink
        (cleanable ?o - object) ; true if the object can be placed in a sink
        (isHot ?o - object) ; true if the object has been heated up
        (heatable ?o - object) ; true if the object can be heated up in a microwave
        (isCool ?o - object) ; true if the object has been cooled
        (coolable ?o - object) ; true if the object can be cooled in the fridge
        (toggleable ?o - object) ; true if the object can be turned on/off
        (isOn ?o - object) ; true if the object is on
        (isToggled ?o - object) ; true if the object has been toggled
        (sliceable ?o - object) ; true if the object can be sliced
        (isSliced ?o - object) ; true if the object is sliced
        (wasInReceptacle ?o - object ?r - receptacle)
    )

    ;; All actions are specified such that the final arguments are the ones used
    ;; for performing actions in Unity.

    ;; agent goes to receptacle
    (:action GotoLocation
        :parameters (?a - agent ?lStart - location ?lEnd - location)
        :precondition (and
            (atLocation ?a ?lStart)
        )
        :effect (and
            (atLocation ?a ?lEnd)
            (not (atLocation ?a ?lStart))
        )
    )

    ;; agent opens receptacle
    (:action OpenObject
        :parameters (?a - agent ?l - location ?r - receptacle)
        :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            ; (forall
            ;     (?re - receptacle)
            ;     (not (opened ?re))
            ; )
        )
        :effect (and
            (opened ?r)
        )
    )
    ;; agent closes receptacle
    (:action CloseObject
        :parameters (?a - agent ?al - location ?r - receptacle)
        :precondition (and
            (atLocation ?a ?al)
            (receptacleAtLocation ?r ?al)
            (openable ?r)
            (opened ?r)
        )
        :effect (and
            (not (opened ?r))
        )

    )

    ;; agent picks up object in a receptacle
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

    ;; agent picks up object not in a receptacle
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

    ;; agent puts down an object in a receptacle
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

    ;; agent puts down an object
    (:action PutObjectInReceptacleObject
        :parameters (?a - agent ?l - location ?ot - otype ?o - object ?outerO - object ?outerR - receptacle)
        :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?outerO ?l)
            (isReceptacleObject ?outerO)
            (not (isReceptacleObject ?o))
            (objectType ?o ?ot)
            (holds ?a ?o)
            (not (holdsAnyReceptacleObject ?a))
            (inReceptacle ?outerO ?outerR)
        )
        :effect (and
            (inReceptacleObject ?o ?outerO)
            (inReceptacle ?o ?outerR)
            (not (holds ?a ?o))
            (not (holdsAny ?a))
            (objectAtLocation ?o ?l)
        )
    )

    ;; agent cleans some object - currently requires it to be in a sink.
    (:action CleanObject
        :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
        :precondition (and
            (receptacleType ?r SinkBasinType)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
        )
        :effect (and
            (isClean ?o)
        )
    )

    ;; agent heats-up some object
    (:action HeatObject
        :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
        :precondition (and
            (receptacleType ?r MicrowaveType)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
        )
        :effect (and
            (isHot ?o)
        )
    )

    ;; agent cools some object
    (:action CoolObject
        :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
        :precondition (and
            (receptacleType ?r FridgeType)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
        )
        :effect (and
            (isCool ?o)
        )
    )

    ;; agent toggle object
    (:action ToggleObject
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

    ;; agent slices some object with a knife
    (:action SliceObject
        :parameters (?a - agent ?l - location ?co - object ?ko - object)
        :precondition (and
            (objectType ?ko KnifeType)
            (atLocation ?a ?l)
            (objectAtLocation ?co ?l)
            (sliceable ?co)
            (holds ?a ?ko)
        )
        :effect (and
            (isSliced ?co)
        )
    )

)