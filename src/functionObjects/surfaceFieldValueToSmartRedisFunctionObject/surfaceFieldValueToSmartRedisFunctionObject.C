/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "surfaceFieldValueToSmartRedisFunctionObject.H"
#include "Time.H"
#include "addToRunTimeSelectionTable.H"
#include "smartRedisClient.H"
#include "surfaceFieldValue.H"
#include "functionObjectList.H"
#include "functionObject.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(surfaceFieldValueToSmartRedisFunctionObject, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        surfaceFieldValueToSmartRedisFunctionObject,
        dictionary
    );
}

}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::surfaceFieldValueToSmartRedisFunctionObject::surfaceFieldValueToSmartRedisFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    smartRedisClient(name, runTime, dict),
    surfaceFVFunctionObjects_(dict.lookup("surfaceFieldValues")),
{}

void Foam::functionObjects::surfaceFieldValueToSmartRedisFunctionObject::_checkNames(functionObjectList& fOL)
{
    forAll(surfaceFVFunctionObjects, i)
    {
       if (fOL.findObjectID(i) == -1)
       {
        FatalErrorInFunction << "Function Object: " << i
            << "was not found in the function object list. "
            << "Make sure it was defined in the controlDict."
            << nl << exit(FatalError);
       }
       // fO->type can be used to query the actual type of the functionObject
    }
}

bool
Foam::functionObjects::surfaceFieldValueToSmartRedisFunctionObject::execute()
{
    Info<< "Writing surfaceFieldValues to SmartRedis database\n" << endl;

    functionObjectList fOL
    (
        IOobject
        (
            "controlDict",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        ),
        dict_
    );

    _checkNames(fOL);
    updateNamingConventionState();
    forAll(surfaceFVFunctionObjects_, i)
    {
        autoPtr<surfaceFieldValue> surfaceFVFunctionObject =
            fOL.findObject(i);
        sendSurfaceFieldValueFields(surfaceFVFunctionObject);
    }
    return true;
}


bool
Foam::functionObjects::surfaceFieldValueToSmartRedisFunctionObject::write()
{
    return true;
}

bool
Foam::functionObjects::surfaceFieldValueToSmartRedisFunctionObject::end()
{
    DataSet ds = getMetadata();
    ds.add_meta_string("EndTimeIndex", Foam::name(mesh().time().timeIndex()));
    client().put_dataset(ds);
    return true;
}

// ************************************************************************* //
