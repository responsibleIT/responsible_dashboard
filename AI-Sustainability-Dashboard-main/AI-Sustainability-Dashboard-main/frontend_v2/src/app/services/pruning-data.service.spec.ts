import { TestBed } from '@angular/core/testing';

import { PruningDataService } from './pruning-data.service';

describe('PruningDataService', () => {
  let service: PruningDataService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PruningDataService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
